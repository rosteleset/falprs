#include <filesystem>
#include <fstream>

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <http_client.h>
#include <opencv2/core/simd_intrinsics.hpp>
#include <userver/clients/http/component.hpp>
#include <userver/engine/sleep.hpp>
#include <userver/formats/serialize/common_containers.hpp>
#include <userver/fs/write.hpp>
#include <userver/http/common_headers.hpp>
#include <userver/http/content_type.hpp>
#include <userver/yaml_config/merge_schemas.hpp>

#include "frs_api.hpp"
#include "frs_workflow.hpp"

namespace tc = triton::client;

namespace Frs
{
  double cosineDistance(const FaceDescriptor& fd1, const FaceDescriptor& fd2)
  {
    if (fd1.cols != fd2.cols || fd1.cols == 0)
      return -1.0;

    constexpr int step = cv::v_float32::nlanes;
    cv::v_float32 sum0 = cv::vx_setzero_f32();
    cv::v_float32 sum1 = cv::vx_setzero_f32();
    cv::v_float32 sum2 = cv::vx_setzero_f32();
    cv::v_float32 sum3 = cv::vx_setzero_f32();
    const auto ptr1 = fd1.ptr<float>(0);
    const auto ptr2 = fd2.ptr<float>(0);
    for (int i = 0; i < fd1.cols; i += 4 * step)
    {
      cv::v_float32 v1 = cv::vx_load(ptr1 + i + 0 * step);
      cv::v_float32 v2 = cv::vx_load(ptr2 + i + 0 * step);
      sum0 += v1 * v2;

      v1 = cv::vx_load(ptr1 + i + 1 * step);
      v2 = cv::vx_load(ptr2 + i + 1 * step);
      sum1 += v1 * v2;

      v1 = cv::vx_load(ptr1 + i + 2 * step);
      v2 = cv::vx_load(ptr2 + i + 2 * step);
      sum2 += v1 * v2;

      v1 = cv::vx_load(ptr1 + i + 3 * step);
      v2 = cv::vx_load(ptr2 + i + 3 * step);
      sum3 += v1 * v2;
    }
    sum0 += sum1 + sum2 + sum3;
    return v_reduce_sum(sum0);
  }

  // OpenCV port of 'LAPV' algorithm (Pech2000)
  double varianceOfLaplacian(const cv::Mat& src)
  {
    cv::Mat lap;
    Laplacian(src, lap, CV_64F);
    constexpr int mrg = 3;  // cut off the border
    lap = lap(cv::Rect(mrg, mrg, lap.cols - 2 * mrg, lap.rows - 2 * mrg));

    cv::Scalar mu, sigma;
    meanStdDev(lap, mu, sigma);

    const double focusMeasure = sigma.val[0] * sigma.val[0];
    return focusMeasure;
  }

  double dist(const double x1, const double y1, const double x2, const double y2)
  {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
  }

  bool isFrontalFace(const cv::Mat& landmarks)
  {
    if (landmarks.empty())
      return false;

    // the nose should be between the eyes
    const float nx = landmarks.at<float>(2, 0);
    const float ny = landmarks.at<float>(2, 1);
    if (nx <= landmarks.at<float>(0, 0) || nx >= landmarks.at<float>(1, 0)
        || ny <= landmarks.at<float>(0, 1) || ny <= landmarks.at<float>(1, 1))
      return false;

    // the right eye should not be to the right of the left tip of the lips
    // the left eye should not be to the left of the right tip of the lips
    if (landmarks.at<float>(0, 0) >= landmarks.at<float>(4, 0)
        || landmarks.at<float>(1, 0) <= landmarks.at<float>(3, 0))
      return false;
    bool is_frontal = true;
    constexpr double equal_threshold = 0.62;  // threshold for equality of two lengths

    // distance from right eye to nose
    double d1 = dist(landmarks.at<float>(0, 0), landmarks.at<float>(0, 1), nx, ny);

    // distance from left eye to nose
    double d2 = dist(landmarks.at<float>(1, 0), landmarks.at<float>(1, 1), nx, ny);
    is_frontal = is_frontal && (std::min(d1, d2) / std::max(d1, d2) > equal_threshold);

    // distance from the right tip of the lips to the nose
    d1 = dist(landmarks.at<float>(3, 0), landmarks.at<float>(3, 1), nx, ny);

    // distance from the left tip of the lips to the nose
    d2 = dist(landmarks.at<float>(4, 0), landmarks.at<float>(4, 1), nx, ny);
    is_frontal = is_frontal && (std::min(d1, d2) / std::max(d1, d2) > equal_threshold);

    // distance from the right tip of the lips to the right eye
    d1 = dist(landmarks.at<float>(3, 0), landmarks.at<float>(3, 1), landmarks.at<float>(0, 0), landmarks.at<float>(0, 1));

    // distance from the left tip of the lips to the left eye
    d2 = dist(landmarks.at<float>(4, 0), landmarks.at<float>(4, 1), landmarks.at<float>(1, 0), landmarks.at<float>(1, 1));
    is_frontal = is_frontal && (std::min(d1, d2) / std::max(d1, d2) > equal_threshold);

    // distance between eyes
    d1 = dist(landmarks.at<float>(0, 0), landmarks.at<float>(0, 1), landmarks.at<float>(1, 0), landmarks.at<float>(1, 1));

    // distance between the tips of the lips
    d2 = dist(landmarks.at<float>(3, 0), landmarks.at<float>(3, 1), landmarks.at<float>(4, 0), landmarks.at<float>(4, 1));

    // distance between the right eye and the right tip of the lips
    const double d3 = dist(landmarks.at<float>(0, 0), landmarks.at<float>(0, 1), landmarks.at<float>(3, 0), landmarks.at<float>(3, 1));

    // distance between the left eye and the left tip of the lips
    const double d4 = dist(landmarks.at<float>(1, 0), landmarks.at<float>(1, 1), landmarks.at<float>(4, 0), landmarks.at<float>(4, 1));

    d1 = std::max(d1, d2);
    d2 = std::max(d3, d4);
    is_frontal = is_frontal && (std::min(d1, d2) / std::max(d1, d2) > equal_threshold);

    return is_frontal;
  }

  // intersection over union
  inline float iou(const float* lbox, const float* rbox)
  {
    const float interBox[] =
      {
        std::max(lbox[0], rbox[0]),  // left
        std::min(lbox[2], rbox[2]),  // right
        std::max(lbox[1], rbox[1]),  // top
        std::min(lbox[3], rbox[3]),  // bottom
      };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
      return 0.0f;

    const float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS + 0.000001f);
  }

  // non maximum suppression algorithm
  inline void nms(std::vector<FaceDetection>& dets, const float nms_thresh = 0.4)
  {
    std::ranges::sort(dets, [](const auto& a, const auto& b)
      { return a.face_confidence > b.face_confidence; });
    for (size_t m = 0; m < dets.size(); ++m)
    {
      auto& [bbox, face_confidence, landmark] = dets[m];
      for (size_t n = m + 1; n < dets.size(); ++n)
      {
        if (iou(bbox, dets[n].bbox) > nms_thresh)
        {
          dets.erase(dets.begin() + static_cast<int>(n));
          --n;
        }
      }
    }
  }

  // face detection area alignment
  cv::Mat alignFaceAffineTransform(const cv::Mat& frame, const cv::Mat& src, const int face_width, const int face_height)
  {
    const cv::Mat dst = (cv::Mat_<double>(5, 2) << 38.2946 / 112.0 * face_width, 51.6963 / 112.0 * face_height,
      73.5318 / 112.0 * face_width, 51.5014 / 112.0 * face_height,
      56.0252 / 112.0 * face_width, 71.7366 / 112.0 * face_height,
      41.5493 / 112.0 * face_width, 92.3655 / 112.0 * face_height,
      70.7299 / 112.0 * face_width, 92.2041 / 112.0 * face_height);
    cv::Mat inliers;
    const cv::Mat mm = cv::estimateAffinePartial2D(src, dst, inliers, cv::LMEDS);
    cv::Mat r;
    warpAffine(frame, r, mm, cv::Size(face_width, face_height));
    return r;
  }

  std::vector<FaceClass> softMax(const std::vector<float>& v)
  {
    std::vector<FaceClass> r(v.size());
    float s = 0.0f;
    for (const float i : v)
      s += exp(i);
    for (size_t i = 0; i < v.size(); ++i)
    {
      r[i].class_index = static_cast<int>(i);
      r[i].score = exp(v[i]) / s;
    }
    std::ranges::sort(r, [](const auto& left, const auto& right)
      { return left.score > right.score; });
    return r;
  }

  cv::Rect enlargeFaceRect(const cv::Rect& face_rect, const double k)
  {
    const int cx = face_rect.tl().x + face_rect.width / 2;
    const int cy = face_rect.tl().y + face_rect.height / 2;
    int max_side = static_cast<int>(std::max(face_rect.width, face_rect.height) * k);
    return {cx - max_side / 2, cy - max_side / 2, max_side, max_side};
  }

  void removeExpiredUnknownDescriptors(std::vector<UnknownDescriptorData>& ud)
  {
    std::erase_if(ud, [now = std::chrono::steady_clock::now()](const auto& item)
      {
        return item.expiration_tp < now;
      });
  }

  Workflow::Workflow(const userver::components::ComponentConfig& config, const userver::components::ComponentContext& context)
    : LoggableComponentBase(config, context),
      task_processor_(context.GetTaskProcessor(config["task_processor"].As<std::string>())),
      fs_task_processor_(context.GetTaskProcessor(config["fs-task-processor"].As<std::string>())),
      http_client_(context.FindComponent<userver::components::HttpClient>().GetHttpClient()),
      logger_(context.FindComponent<userver::components::Logging>().GetLogger(std::string(kLogger))),
      pg_cluster_(context.FindComponent<userver::components::Postgres>(kDatabase).GetCluster()),
      common_config_cache_(context.FindComponent<ConfigCache>()),
      vstreams_config_cache_(context.FindComponent<VStreamsConfigCache>()),
      face_descriptor_cache_(context.FindComponent<FaceDescriptorCache>()),
      vstream_descriptors_cache_(context.FindComponent<VStreamDescriptorsCache>()),
      sg_config_cache_(context.FindComponent<SGConfigCache>()),
      sg_descriptors_cache_(context.FindComponent<SGDescriptorsCache>())
  {
    local_config_.allow_group_id_without_auth = config[ConfigParams::SECTION_NAME][ConfigParams::ALLOW_GROUP_ID_WITHOUT_AUTH].As<decltype(local_config_.allow_group_id_without_auth)>();

    local_config_.screenshots_path = config[ConfigParams::SECTION_NAME][ConfigParams::SCREENSHOTS_PATH].As<decltype(local_config_.screenshots_path)>();
    // make sure the path ends with /
    if (!local_config_.screenshots_path.empty() && !local_config_.screenshots_path.ends_with('/'))
      local_config_.screenshots_path = local_config_.screenshots_path + '/';

    local_config_.screenshots_url_prefix = config[ConfigParams::SECTION_NAME][ConfigParams::SCREENSHOTS_URL_PREFIX].As<decltype(local_config_.screenshots_url_prefix)>();
    // make sure the path ends with /
    if (!local_config_.screenshots_url_prefix.empty() && !local_config_.screenshots_url_prefix.ends_with("/"))
      local_config_.screenshots_url_prefix += "/";

    local_config_.events_path = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_PATH].As<decltype(local_config_.events_path)>();
    // make sure the path ends with /
    if (!local_config_.events_path.empty() && !local_config_.events_path.ends_with('/'))
      local_config_.events_path = local_config_.events_path + '/';

    local_config_.clear_old_log_faces = config[ConfigParams::SECTION_NAME][ConfigParams::CLEAR_OLD_LOG_FACES].As<decltype(local_config_.clear_old_log_faces)>();
    local_config_.log_faces_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::LOG_FACES_TTL].As<decltype(local_config_.log_faces_ttl)>();
    local_config_.flag_deleted_maintenance_interval = config[ConfigParams::SECTION_NAME][ConfigParams::FLAG_DELETED_MAINTENANCE_INTERVAL].As<decltype(local_config_.flag_deleted_maintenance_interval)>();
    local_config_.flag_deleted_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::FLAG_DELETED_TTL].As<decltype(local_config_.flag_deleted_ttl)>();
    local_config_.copy_events_maintenance_interval = config[ConfigParams::SECTION_NAME][ConfigParams::COPY_EVENTS_MAINTENANCE_INTERVAL].As<decltype(local_config_.copy_events_maintenance_interval)>();
    local_config_.clear_old_events = config[ConfigParams::SECTION_NAME][ConfigParams::CLEAR_OLD_EVENTS].As<decltype(local_config_.clear_old_events)>();
    local_config_.events_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_TTL].As<decltype(local_config_.events_ttl)>();

    loadDNNStatsData();

    // Periodic maintenance
    if (local_config_.clear_old_log_faces.count() > 0)
      old_logs_maintenance_task_.Start(kOldLogsMaintenance,
        {local_config_.clear_old_log_faces, {userver::utils::PeriodicTask::Flags::kStrong}},
        [this]
        {
          doOldLogMaintenance();
        });

    if (local_config_.flag_deleted_maintenance_interval.count() > 0)
      flag_deleted_maintenance_task_.Start(kFlagDeletedMaintenance,
        {local_config_.flag_deleted_maintenance_interval, {userver::utils::PeriodicTask::Flags::kStrong}},
        [this]
        {
          doFlagDeletedMaintenance();
        });

    if (local_config_.copy_events_maintenance_interval.count() > 0)
      copy_events_maintenance_task_.Start(kCopyEventsMaintenance,
        {local_config_.copy_events_maintenance_interval, {userver::utils::PeriodicTask::Flags::kStrong}},
        [this]
        {
          doCopyEventsMaintenance();
        });

    if (local_config_.clear_old_events.count() > 0)
      old_events_maintenance_task_.Start(kOldLogsMaintenance,
        {local_config_.clear_old_events, {userver::utils::PeriodicTask::Flags::kStrong}},
        [this]
        {
          doOldEventsMaintenance();
        });
  }

  Workflow::~Workflow()
  {
    saveDNNStatsData();
  }

  userver::yaml_config::Schema Workflow::GetStaticConfigSchema()
  {
    return userver::yaml_config::MergeSchemas<LoggableComponentBase>(R"~(
# yaml
type: object
description: Component for face recognition workflow
additionalProperties: false
properties:
    task_processor:
        type: string
        description: main task processor for recognition workflow
    fs-task-processor:
        type: string
        description: task processor to process filesystem bound tasks
    config:
        type: object
        description: default configuration parameters
        additionalProperties: false
        properties:
            allow-group-id-without-auth:
                type: number
                description: Allow use of a group with a specified identifier without authorization
                defaultDescription: 1
            screenshots-path:
                type: string
                description: Local path for saving faces screenshots
                defaultDescription: '/opt/falprs/static/frs/screenshots/'
            screenshots-url-prefix:
                type: string
                description: Web URL prefix for faces screenshots
                defaultDescription: 'http://localhost:9051/frs/screenshots/'
            events-path:
                type: string
                description: Local path for saving events screenshots
                defaultDescription: '/opt/falprs/static/frs/events/'
            clear-old-log-faces:
                type: string
                description: Period for launching cleaning of outdated logs from the log_faces table
                defaultDescription: 4h
            flag-deleted-maintenance-interval:
                type: string
                description: Maintenance period for records marked for deletion
                defaultDescription: 10s
            flag-deleted-ttl:
                type: string
                description: TTL of records marked for deletion
                defaultDescription: 5m
            copy-events-maintenance-interval:
                type: string
                description: Event data copy maintenance period
                defaultDescription: 30s
            clear-old-events:
                type: string
                description: Period for launching cleaning of outdated events
                defaultDescription: 1d
            log-faces-ttl:
                type: string
                description: TTL of logs from the log_faces table
                defaultDescription: 4h
            events-ttl:
                type: string
                description: TTL of the copied events
                defaultDescription: 30d
  )~");
  }

  const userver::logging::LoggerPtr& Workflow::getLogger() const
  {
    return logger_;
  }

  const LocalConfig& Workflow::getLocalConfig() const
  {
    return local_config_;
  }

  void Workflow::startWorkflow(std::string&& vstream_key)
  {
    int32_t id_group = -1;
    std::chrono::milliseconds workflow_timeout{std::chrono::seconds{0}};
    // scope for accessing cache
    {
      const auto cache = vstreams_config_cache_.Get();
      if (!cache->getData().contains(vstream_key))
        return;
      id_group = cache->getData().at(vstream_key).id_group;
      workflow_timeout = cache->getData().at(vstream_key).workflow_timeout;
    }
    if (id_group <= 0)
      return;

    bool do_pipeline = false;
    // scope for accessing concurrent variable
    {
      auto data_ptr = being_processed_vstreams.Lock();
      if (!data_ptr->contains(vstream_key))
        do_pipeline = true;
      (*data_ptr)[vstream_key] = true;
    }

    if (workflow_timeout.count() > 0)
    {
      auto data_ptr = vstream_timeouts.Lock();
      (*data_ptr)[vstream_key] = std::chrono::steady_clock::now() + workflow_timeout;
    }

    if (do_pipeline)
    {
      TaskData task_data{
        .id_group = id_group,
        .vstream_key = std::move(vstream_key),
        .task_type = TASK_RECOGNIZE,
        .frame_url = {}
      };
      tasks_.Detach(AsyncNoSpan(task_processor_, &Workflow::processPipeline, this, std::move(task_data)));
    }
  }

  void Workflow::stopWorkflow(std::string&& vstream_key, const bool is_internal)
  {
    // scope for accessing concurrent variable
    {
      auto data_ptr = being_processed_vstreams.Lock();
      if (data_ptr->contains(vstream_key))
      {
        if (is_internal)
          data_ptr->erase(vstream_key);
        else
          (*data_ptr)[vstream_key] = false;
      }
    }

    // scope for accessing concurrent variable
    {
      auto data_ptr = vstream_timeouts.Lock();
      if (data_ptr->contains(vstream_key))
        data_ptr->erase(vstream_key);
    }
  }

  DescriptorRegistrationResult Workflow::processPipeline(TaskData&& task_data)
  {
    std::string url;
    CommonConfig common_config{};
    VStreamConfig config{};
    DescriptorRegistrationResult result;

    // scope for accessing cache
    {
      auto cache = common_config_cache_.Get();
      if (cache->getCommonConfig().contains(task_data.id_group))
        common_config = cache->getCommonConfig().at(task_data.id_group);
      if (cache->getDefaultVStreamConfig().contains(task_data.id_group))
        config = cache->getDefaultVStreamConfig().at(task_data.id_group);
    }

    // scope for accessing cache
    {
      auto cache = vstreams_config_cache_.Get();
      if (!task_data.vstream_key.empty() && !cache->getData().contains(task_data.vstream_key))
      {
        stopWorkflow(std::move(task_data.vstream_key));
        return {
          .comments = absl::Substitute("Invalid video stream key: $0", task_data.vstream_key),
          .face_image = {},
          .id_descriptors = {}
        };
      }

      if (!task_data.vstream_key.empty() && cache->getData().contains(task_data.vstream_key))
        config = cache->getData().at(task_data.vstream_key);
      if (task_data.task_type == TASK_RECOGNIZE)
        url = config.url;
      else
        url = task_data.frame_url;
    }

    auto delay_between_frames = config.delay_between_frames;

    if (config.logs_level <= userver::logging::Level::kDebug || task_data.task_type == TASK_TEST)
    {
      auto frame_url = url.starts_with("data:") ? "data:base64..." : url;
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "Start processPipeline: vstream_key = {};  task_type = {};  url = {}",
        task_data.vstream_key, static_cast<int>(task_data.task_type), frame_url);
    }

    try
    {
      std::string image_data;
      if (url.starts_with("data:"))
      {
        if (auto pos_comma = url.find(','); pos_comma != std::string::npos)
          if (url.find(";base64,") != std::string::npos)
            if (!absl::Base64Unescape(absl::ClippedSubstr(url, pos_comma + 1), &image_data))
            {
              if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
                  "Error decoding image from BASE64: vstream_key = {};",
                  task_data.vstream_key);

              return {
                .comments = "Error decoding image from BASE64",
                .face_image = {},
                .id_descriptors = {}
              };
            }
      } else
      {
        if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  before image acquisition",
            task_data.vstream_key);
        auto capture_response = http_client_.CreateRequest()
          .get(url)
          .retry(config.max_capture_error_count)
          .timeout(config.capture_timeout)
          .perform();
        if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  after image acquisition",
            task_data.vstream_key);

        if (capture_response->status_code() != userver::clients::http::Status::OK || capture_response->body_view().empty())
        {
          if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
              "vstream_key = {};  url = {};  status_code = {}",
              task_data.vstream_key, url, capture_response->status_code());
          if (config.delay_after_error.count() > 0)
          {
            if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
                "vstream_key = {};  delay for {}ms",
                task_data.vstream_key, config.delay_after_error.count());
            if (task_data.task_type == TASK_RECOGNIZE)
              nextPipeline(std::move(task_data), config.delay_after_error);
          } else if (task_data.task_type == TASK_RECOGNIZE)
            stopWorkflow(std::move(task_data.vstream_key));

          return {
            .comments = absl::Substitute("Error when retrieving image by url: $0", url),
            .face_image = {},
            .id_descriptors = {}
          };
        }

        image_data = capture_response->body();
      }

      if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      {
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  image size = {} bytes",
          task_data.vstream_key, image_data.size());
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  before decoding the image",
          task_data.vstream_key);
      }
      cv::Mat frame = imdecode(std::vector<char>(image_data.begin(), image_data.end()), cv::IMREAD_COLOR);
      if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  after decoding the image",
          task_data.vstream_key);

      cv::Rect work_area{};
      if (config.work_area.size() == 4)
      {
        work_area.x = static_cast<int>(config.work_area[0] * frame.cols / 100.0f);
        work_area.y = static_cast<int>(config.work_area[1] * frame.rows / 100.0f);
        work_area.width = static_cast<int>(config.work_area[2] * frame.cols / 100.0f);
        work_area.height = static_cast<int>(config.work_area[3] * frame.rows / 100.0f);
      }

      if (task_data.task_type == TASK_REGISTER_DESCRIPTOR)
      {
        // if the face search area is not specified, then we search throughout the entire image
        if (task_data.face_width == 0)
          task_data.face_width = frame.cols;
        if (task_data.face_height == 0)
          task_data.face_height = frame.rows;
      }

      // looking for faces
      if (std::vector<FaceDetection> detected_faces; detectFaces(task_data, frame, config, detected_faces))
      {
        DNNStatsData stats_data;
        ++stats_data.fd_count;
        std::vector<FaceData> face_data;
        int recognized_face_count = 0;
        double best_quality = 0.0;
        int best_face_index = -1;
        double best_register_quality = 0.0;
        double best_register_ioa = 0.0;
        int best_register_index = -1;
        bool has_sgroup_events = false;

        if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  process the found faces, quantity: {}",
            task_data.vstream_key, detected_faces.size());
        for (auto& [bbox, face_confidence, landmark] : detected_faces)
        {
          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  face probability: {:.3f}",
              task_data.vstream_key, face_confidence);
          auto work_region = cv::Rect(
            static_cast<int>(config.margin / 100.0 * frame.cols),
            static_cast<int>(config.margin / 100.0 * frame.rows),
            static_cast<int>(frame.cols - 2.0 * frame.cols * config.margin / 100.0),
            static_cast<int>(frame.rows - 2.0 * frame.rows * config.margin / 100.0));
          if (!work_area.empty())
            work_region = work_region & work_area;
          auto face_rect = cv::Rect(
            static_cast<int>(bbox[0]),
            static_cast<int>(bbox[1]),
            static_cast<int>(bbox[2] - bbox[0] + 1),
            static_cast<int>(bbox[3] - bbox[1] + 1));

          face_data.emplace_back();
          face_data.back().face_rect = face_rect;

          // check that the face is completely in the work area
          if ((work_region & face_rect) != face_rect)
          {
            if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  the person is not in the work area",
                task_data.vstream_key);
            continue;
          }

          face_data.back().is_work_area = true;
          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  the person is in the work area",
              task_data.vstream_key);

          auto landmarks5 = cv::Mat(5, 2, CV_32F, landmark);
          face_data.back().landmarks5 = landmarks5.clone();

          // checking the frontality of the face using markers
          if (!isFrontalFace(landmarks5))
          {
            if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  face is not frontal according to markers",
                task_data.vstream_key);
            continue;
          }

          // face "alignment" for face recognition inference
          cv::Mat aligned_face = alignFaceAffineTransform(frame, landmarks5, common_config.dnn_fr_input_width, common_config.dnn_fr_input_height);
          if (aligned_face.cols != common_config.dnn_fr_input_width || aligned_face.rows != common_config.dnn_fr_input_height)
          {
            if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  failed to do face alignment to get descriptor",
                task_data.vstream_key);
            continue;
          }

          if (task_data.task_type == TASK_TEST)
            AsyncNoSpan(fs_task_processor_,
              [&]
              {
                cv::imwrite(absl::Substitute("$0/aligned_face_$1.jpg", std::filesystem::current_path().string(), face_data.size()), aligned_face);
              }).Get();

          face_data.back().is_frontal = true;

          // check for blur
          auto laplacian = varianceOfLaplacian(aligned_face);
          face_data.back().laplacian = laplacian;
          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  laplacian = {:.2f}",
              task_data.vstream_key, laplacian);
          if (laplacian < config.blur || laplacian > config.blur_max)
          {
            if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  the face is blurry or too clear",
                task_data.vstream_key);

            continue;
          }
          face_data.back().is_non_blurry = true;
          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  the face is not blurry",
              task_data.vstream_key);

          // face "alignment" for face class inference
          auto aligned_face_class = alignFaceAffineTransform(frame, landmarks5, common_config.dnn_fc_input_width, common_config.dnn_fc_input_height);
          if (aligned_face_class.cols != common_config.dnn_fc_input_width || aligned_face_class.rows != common_config.dnn_fc_input_height)
          {
            if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  failed to do face alignment for inference class",
                task_data.vstream_key);

            continue;
          }
          if (task_data.task_type == TASK_TEST)
            AsyncNoSpan(fs_task_processor_,
              [&]
              {
                cv::imwrite(absl::Substitute("$0/aligned_face_class_$1.jpg", std::filesystem::current_path().string(), face_data.size()), aligned_face_class);
              }).Get();

          // checking the class of the face (normal, wearing a mask, wearing sunglasses)
          if (std::vector<FaceClass> face_classes; inferFaceClass(task_data, aligned_face_class, config, face_classes))
          {
            ++stats_data.fc_count;
            face_data.back().face_class_index = static_cast<FaceClassIndexes>(face_classes[0].class_index);
            face_data.back().face_class_confidence = face_classes[0].score;
            if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  face class: {};  probability: {:.3f}",
                task_data.vstream_key, face_classes[0].class_index, face_classes[0].score);
          }
          if (face_data.back().face_class_index == FACE_NONE
              || (face_data.back().face_class_index != FACE_NORMAL
                  && face_data.back().face_class_confidence > config.face_class_confidence))
            continue;

          face_data.back().face_class_index = FACE_NORMAL;

          if (task_data.task_type == TASK_REGISTER_DESCRIPTOR)
          {
            cv::Rect r(task_data.face_left, task_data.face_top, task_data.face_width, task_data.face_height);
            auto f_intersection = (r & face_data.back().face_rect).area();
            if (auto f_area = face_data.back().face_rect.area(); f_area > 0)
              face_data.back().ioa = static_cast<double>(f_intersection) / static_cast<double>(f_area);
          }

          // get a facial descriptor (biometric template)
          if (bool infer_face_descriptor_result = extractFaceDescriptor(task_data, aligned_face, config, face_data.back().fd); !infer_face_descriptor_result)
            continue;
          ++stats_data.fr_count;
          auto face_descriptor = face_data.back().fd.clone();
          double norm_l2 = cv::norm(face_descriptor, cv::NORM_L2);
          if (norm_l2 <= 0.0)
            norm_l2 = 1.0;
          face_descriptor = face_descriptor / norm_l2;

          // recognize the face
          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  before recognition",
              task_data.vstream_key);
          double max_cos_distance = -2.0;
          int id_descriptor{};

          // scope for accessing cache
          {
            auto vd_cache = vstream_descriptors_cache_.Get();
            auto fd_cache = face_descriptor_cache_.Get();
            auto sgc_cache = sg_config_cache_.Get();
            auto sgd_cache = sg_descriptors_cache_.Get();

            if (config.id_vstream > 0 && vd_cache->getData().contains(config.id_vstream))
              for (const auto& item : vd_cache->getData().at(config.id_vstream))
                if (fd_cache->getData().contains(item))
                {
                  if (double cos_distance = cosineDistance(face_descriptor, fd_cache->getData().at(item)); cos_distance > max_cos_distance)
                  {
                    max_cos_distance = cos_distance;
                    id_descriptor = item;
                  }
                }
            if (fd_cache->getSpawned().contains(id_descriptor))
            {
              auto id_parent = fd_cache->getSpawned().at(id_descriptor);
              if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                  "vstream_key = {};  spawned id_descriptor = {};  parent id_descriptor = {}",
                  task_data.vstream_key, id_descriptor, id_parent);
              id_descriptor = id_parent;
            }

            // recognition in special groups
            if (task_data.id_sgroup > 0)
            {
              if (sgd_cache->getData().contains(task_data.id_sgroup))
                for (const auto& id_sg_descriptor : sgd_cache->getData().at(task_data.id_sgroup))
                  if (fd_cache->getData().contains(id_sg_descriptor))
                  {
                    if (double cos_distance = cosineDistance(face_descriptor, fd_cache->getData().at(id_sg_descriptor)); cos_distance > max_cos_distance)
                    {
                      max_cos_distance = cos_distance;
                      id_descriptor = id_sg_descriptor;
                    }
                  }
            } else
            {
              if (sgc_cache->getMappedSG().contains(config.id_group))
                for (const auto& id_sgroup : sgc_cache->getMappedSG().at(config.id_group))
                  if (sgd_cache->getData().contains(id_sgroup))
                  {
                    double sg_max_cos_distance = -2.0;
                    int id_sg_best_descriptor{};
                    for (const auto& id_sg_descriptor : sgd_cache->getData().at(id_sgroup))
                      if (fd_cache->getData().contains(id_sg_descriptor))
                      {
                        if (double cos_distance = cosineDistance(face_descriptor, fd_cache->getData().at(id_sg_descriptor)); cos_distance > sg_max_cos_distance)
                        {
                          sg_max_cos_distance = cos_distance;
                          id_sg_best_descriptor = id_sg_descriptor;
                        }
                      }
                    if (id_sg_best_descriptor > 0 && sg_max_cos_distance >= config.tolerance)
                    {
                      face_data.back().sg_descriptors[id_sgroup] = {sg_max_cos_distance, id_sg_best_descriptor};
                      has_sgroup_events = true;
                    }
                  }
            }
          }

          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  after recognition",
              task_data.vstream_key);

          face_data.back().cosine_distance = max_cos_distance;

          if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
              "vstream_key = {};  most similar data: cosine_distance = {:.3f};  id_descriptor = {}",
              task_data.vstream_key, max_cos_distance, id_descriptor);

          if (id_descriptor == 0 || max_cos_distance < config.tolerance)
          {
            // face not recognized
            if (face_data.back().laplacian > best_quality && recognized_face_count == 0)
            {
              best_quality = face_data.back().laplacian;
              best_face_index = static_cast<int>(face_data.size()) - 1;
            }

            if (config.flag_spawned_descriptors && task_data.task_type == TASK_RECOGNIZE)
            {
              if (config.logs_level <= userver::logging::Level::kTrace)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                  "vstream_key = {};  add an unknown descriptor",
                  task_data.vstream_key);

              auto ud_ptr = unknown_descriptors.Lock();
              removeExpiredUnknownDescriptors((*ud_ptr)[config.id_vstream]);

              // add an unknown descriptor
              if (!ud_ptr->contains(config.id_vstream))
                (*ud_ptr)[config.id_vstream] = {};
              cv::Rect r = enlargeFaceRect(face_data.back().face_rect, config.face_enlarge_scale);
              r = r & cv::Rect(0, 0, frame.cols, frame.rows);
              (*ud_ptr)[config.id_vstream].emplace_back(std::chrono::steady_clock::now() + config.unknown_descriptor_ttl,
                face_data.back().fd.clone(), frame(r).clone());
            }
          } else
          {
            // face recognized
            face_data.back().id_descriptor = id_descriptor;
            ++recognized_face_count;

            if (recognized_face_count == 1 || face_data.back().laplacian > best_quality)
            {
              best_quality = face_data.back().laplacian;
              best_face_index = static_cast<int>(face_data.size()) - 1;
            }

            if (task_data.task_type == TASK_PROCESS_FRAME)
              result.id_descriptors.push_back(id_descriptor);

            if (config.flag_spawned_descriptors && task_data.task_type == TASK_RECOGNIZE)
            {
              FaceDescriptor fd_spawned;
              cv::Mat face_image_spawned;

              // scope for accessing concurrent variable
              {
                auto ud_ptr = unknown_descriptors.Lock();
                removeExpiredUnknownDescriptors((*ud_ptr)[config.id_vstream]);

                // find and create a spawned descriptor among the unknowns if necessary
                double max_cd = -2.0;
                auto k = (*ud_ptr)[config.id_vstream].size();
                for (size_t i = 0; i < (*ud_ptr)[config.id_vstream].size(); ++i)
                {
                  auto fd = (*ud_ptr)[config.id_vstream][i].fd.clone();
                  double n_l2 = cv::norm(fd, cv::NORM_L2);
                  if (n_l2 <= 0.0)
                    n_l2 = 1.0;
                  fd = fd / n_l2;
                  if (double cos_distance = cosineDistance(face_descriptor, fd); cos_distance > max_cd)
                  {
                    max_cd = cos_distance;
                    k = i;
                  }
                }

                if (k < (*ud_ptr)[config.id_vstream].size() && max_cd > config.tolerance)
                {
                  if (config.logs_level <= userver::logging::Level::kTrace)
                    USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                      "vstream_key = {};  unknown descriptors cosine distance = {:.3f};  index = {};  size = {}",
                      task_data.vstream_key, max_cd, k, (*ud_ptr)[config.id_vstream].size());
                  fd_spawned = std::move((*ud_ptr)[config.id_vstream][k].fd);
                  face_image_spawned = std::move((*ud_ptr)[config.id_vstream][k].face_image);
                }

                // clear the unknown descriptors anyway
                (*ud_ptr)[config.id_vstream].clear();
              }

              if (!fd_spawned.empty())
              {
                auto id_spawned = addFaceDescriptor(config.id_group, config.id_vstream, fd_spawned, face_image_spawned, id_descriptor);
                if (config.logs_level <= userver::logging::Level::kTrace)
                  USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                    "vstream_key = {};  created spawned descriptor with id = {};  id_parent = {}",
                    task_data.vstream_key, id_spawned, id_descriptor);
              }
            }
          }

          if (task_data.task_type == TASK_REGISTER_DESCRIPTOR)
          {
            if (face_data.back().ioa > 0.999 && face_data.back().laplacian > best_register_quality)
            {
              best_register_quality = face_data.back().laplacian;
              best_register_index = static_cast<int>(face_data.size()) - 1;
            }
            if (fabs(best_register_quality) < 0.001 && face_data.back().ioa > best_register_ioa)
            {
              best_register_ioa = face_data.back().ioa;
              best_register_index = static_cast<int>(face_data.size()) - 1;
            }
          }
        }  // end of the detected faces loop

        // to collect inference statistics
        {
          auto dnn_stats_ptr = dnn_stats_data.Lock();
          (*dnn_stats_ptr)[config.id_vstream].fd_count += stats_data.fd_count;
          (*dnn_stats_ptr)[config.id_vstream].fc_count += stats_data.fc_count;
          (*dnn_stats_ptr)[config.id_vstream].fr_count += stats_data.fr_count;
        }

        std::string frame_with_osd;
        if (best_face_index >= 0 && task_data.task_type == TASK_RECOGNIZE)
        {
          if (config.logs_level <= userver::logging::Level::kInfo)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
              "vstream_key = {};  faces detected: id_vstream = {}",
              task_data.vstream_key, config.id_vstream);

          // draw, if necessary, OSD
          if (!config.title.empty())
          {
            constexpr int font_face = cv::FONT_HERSHEY_COMPLEX;
            constexpr int thickness = 2;

            auto osd_date = absl::Now();
            auto datetime_text = FormatTime(config.osd_dt_format, osd_date, absl::LocalTimeZone());
            cv::Size text_size = cv::getTextSize(datetime_text, font_face, 1.0, thickness, nullptr);

            // recalculate font_scale and text_size
            const double font_scale = config.title_height_ratio * static_cast<float>(frame.rows) / static_cast<float>(text_size.height);
            text_size = cv::getTextSize(datetime_text, font_face, font_scale, thickness, nullptr);

            // draw date and time in the upper left corner
            // to draw text with a contour, draw it twice with different thicknesses
            putText(frame, datetime_text, {10, 10 + text_size.height}, font_face, font_scale, cv::Scalar::all(0), thickness + 2, cv::LINE_AA);
            putText(frame, datetime_text, {10, 10 + text_size.height}, font_face, font_scale, cv::Scalar::all(255), thickness, cv::LINE_AA);

            // draw the title in the lower left corner
            // to draw text with a contour, draw it twice with different thicknesses
            putText(frame, config.title, {10, frame.rows - 10}, font_face, font_scale, cv::Scalar::all(0), thickness + 2, cv::LINE_AA);
            putText(frame, config.title, {10, frame.rows - 10}, font_face, font_scale, cv::Scalar::all(255), thickness, cv::LINE_AA);

            std::vector<uchar> new_frame;
            imencode(".jpg", frame, new_frame);
            frame_with_osd = std::string(new_frame.begin(), new_frame.end());
          }

          auto log_uuid = boost::uuids::random_generator()();
          auto s_uuid = absl::StrReplaceAll(boost::uuids::to_string(log_uuid), {{"-", ""}});
          auto path_suffix = absl::Substitute("group_$0/$1/$2/$3/$4/", config.id_group, s_uuid[0], s_uuid[1], s_uuid[2], s_uuid[3]);
          auto screenshot_extension = ".jpg";

          auto log_date = userver::storages::postgres::TimePointTz{std::chrono::system_clock::now()};
          auto id_log = addLogFace(config.id_vstream, log_date, face_data[best_face_index].id_descriptor,
            face_data[best_face_index].laplacian, face_data[best_face_index].face_rect,
            absl::StrCat(local_config_.screenshots_url_prefix, path_suffix, s_uuid, screenshot_extension), log_uuid);

          // write screenshot to a file
          auto path_prefix = absl::StrCat(local_config_.screenshots_path, path_suffix);
          userver::fs::CreateDirectories(fs_task_processor_, path_prefix);
          auto path = absl::StrCat(path_prefix, s_uuid, screenshot_extension);
          userver::fs::RewriteFileContents(fs_task_processor_, path, frame_with_osd.empty() ? image_data : frame_with_osd);
          userver::fs::Chmod(fs_task_processor_, path,
            boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write | boost::filesystem::perms::others_read | boost::filesystem::perms::others_write);

          if (id_log > 0 && face_data[best_face_index].id_descriptor > 0 && !config.callback_url.empty())
          {
            // send an event about face recognition
            userver::formats::json::ValueBuilder json_data;
            json_data[Api::P_FACE_ID] = face_data[best_face_index].id_descriptor;
            json_data[Api::P_LOG_EVENT_ID] = id_log;
            DeliveryEventResult delivery_result = ERROR;
            try
            {
              auto delivery_response = http_client_.CreateRequest()
               .post(config.callback_url)
               .headers({{userver::http::headers::kContentType, userver::http::content_type::kApplicationJson.ToString()}})
               .data(ToString(json_data.ExtractValue()))
               .timeout(common_config.callback_timeout)
               .perform();
              delivery_result = (delivery_response->status_code() == userver::clients::http::Status::OK
                || delivery_response->status_code() == userver::clients::http::Status::NoContent) ? SUCCESSFUL : ERROR;
            } catch (const std::exception& e)
            {
              delivery_result = ERROR;
              LOG_ERROR_TO(logger_) << e.what();
            }
            if (delivery_result == SUCCESSFUL)
            {
              if (config.logs_level <= userver::logging::Level::kInfo)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
                  "vstream_key = {};  facial recognition event sent: id_vstream = {}; id_descriptor = {}",
                  task_data.vstream_key, config.id_vstream, face_data[best_face_index].id_descriptor);
            } else
              LOG_ERROR_TO(logger_,
                "vstream_key = {};  error sending facial recognition event data to callback {}",
                task_data.vstream_key, config.callback_url);
          }

          // write event's data to files
          AsyncNoSpan(fs_task_processor_,
            [&]
            {
              std::ofstream ff(absl::StrCat(path_prefix, s_uuid, DATA_FILE_SUFFIX), std::ios::binary);
              userver::formats::json::ValueBuilder json_faces;
              for (size_t i = 0; i < face_data.size(); ++i)
              {
                std::vector<float> landmarks5;
                landmarks5.reserve(10);
                if (!face_data[i].landmarks5.empty())
                  for (int k = 0; k < 5; ++k)
                  {
                    landmarks5.push_back(face_data[i].landmarks5.at<float>(k, 0));
                    landmarks5.push_back(face_data[i].landmarks5.at<float>(k, 1));
                  }
                userver::formats::json::ValueBuilder v;
                v["left"] = face_data[i].face_rect.x;
                v["top"] = face_data[i].face_rect.y;
                v["width"] = face_data[i].face_rect.width;
                v["height"] = face_data[i].face_rect.height;
                v["laplacian"] = face_data[i].laplacian;
                v["landmarks5"] = landmarks5;
                v["face_class_index"] = static_cast<int>(face_data[i].face_class_index);
                v["id_descriptor"] = face_data[i].id_descriptor;
                v["face_class_confidence"] = face_data[i].face_class_confidence;
                v["is_frontal"] = face_data[i].is_frontal;
                v["is_non_blurry"] = face_data[i].is_non_blurry;
                v["is_work_area"] = face_data[i].is_work_area;
                json_faces.PushBack(std::move(v));
                if (!face_data[i].fd.empty())
                {
                  // write the descriptor data to a binary file
                  ff.write(s_uuid.data(), static_cast<std::streamsize>(s_uuid.size()));
                  ff.write(reinterpret_cast<char*>(&i), sizeof(int32_t));
                  ff.write(reinterpret_cast<const char*>(face_data[i].fd.data), static_cast<std::streamsize>(common_config.dnn_fr_output_size * sizeof(float)));
                }
              }

              // write JSON data of the event
              userver::formats::json::ValueBuilder json_data;
              json_data["id_vstream"] = config.id_vstream;
              json_data["event_date"] = log_date;
              json_data["best_face_index"] = best_face_index;
              json_data["faces"] = std::move(json_faces);
              std::ofstream f_json(absl::StrCat(path_prefix, s_uuid, JSON_SUFFIX));
              f_json << ToString(json_data.ExtractValue());
            }).Get();
        }

        // send events about face recognition from special groups
        if (has_sgroup_events && task_data.task_type == TASK_RECOGNIZE)
          for (const auto& [face_rect, is_work_area, is_frontal, is_non_blurry, face_class_index, face_class_confidence, cosine_distance, fd, landmarks5, laplacian, ioa, id_descriptor, sg_descriptors] : face_data)
            for (const auto& [fst, snd] : sg_descriptors)
            {
              auto log_uuid = boost::uuids::random_generator()();
              auto s_uuid = absl::StrReplaceAll(boost::uuids::to_string(log_uuid), {{"-", ""}});
              auto path_suffix = absl::Substitute("group_$0/$1/$2/$3/$4/", config.id_group, s_uuid[0], s_uuid[1], s_uuid[2], s_uuid[3]);
              auto screenshot_extension = ".jpg";
              auto screenshot_url = absl::StrCat(local_config_.screenshots_url_prefix, path_suffix, s_uuid, screenshot_extension);

              auto log_date = userver::storages::postgres::TimePointTz{std::chrono::system_clock::now()};
              auto id_log = addLogFace(config.id_vstream, log_date, snd.id_descriptor, laplacian, face_rect, screenshot_url, log_uuid, DISABLED);

              // write screenshot to a file
              auto path_prefix = absl::StrCat(local_config_.screenshots_path, path_suffix);
              userver::fs::CreateDirectories(fs_task_processor_, path_prefix);
              auto path = absl::StrCat(path_prefix, s_uuid, screenshot_extension);
              userver::fs::RewriteFileContents(fs_task_processor_, path, frame_with_osd.empty() ? image_data : frame_with_osd);
              userver::fs::Chmod(fs_task_processor_, path,
                boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write | boost::filesystem::perms::others_read | boost::filesystem::perms::others_write);

              std::string sg_group_callback_url;
              // scope for accessing cache
              {
                if (auto sg_config = sg_config_cache_.Get(); sg_config->getMap().contains(fst))
                  sg_group_callback_url = sg_config->getData().at(sg_config->getMap().at(fst)).callback_url;
              }

              if (id_log > 0 && !sg_group_callback_url.empty())
              {
                // send data to callback
                userver::formats::json::ValueBuilder json_data;
                json_data[Api::P_FACE_ID] = snd.id_descriptor;
                json_data[Api::P_SCREENSHOT_URL] = screenshot_url;
                json_data[Api::P_DATE] = log_date;
                DeliveryEventResult delivery_result = ERROR;
                try
                {
                  auto delivery_response = http_client_.CreateRequest()
                   .post(sg_group_callback_url)
                   .headers({{userver::http::headers::kContentType, userver::http::content_type::kApplicationJson.ToString()}})
                   .data(ToString(json_data.ExtractValue()))
                   .timeout(common_config.callback_timeout)
                   .perform();
                  delivery_result = (delivery_response->status_code() == userver::clients::http::Status::OK
                    || delivery_response->status_code() == userver::clients::http::Status::NoContent) ? SUCCESSFUL : ERROR;
                } catch (const std::exception& e)
                {
                  delivery_result = ERROR;
                  LOG_ERROR_TO(logger_) << e.what();
                }
                if (delivery_result == SUCCESSFUL)
                {
                  if (config.logs_level <= userver::logging::Level::kInfo)
                    USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
                      "vstream_key = {};  an event was sent about facial recognition in a special group: id_sgroup = {}; id_vstream = {}; id_descriptor = {}",
                      task_data.vstream_key, fst, config.id_vstream, snd.id_descriptor);
                } else
                  LOG_ERROR_TO(logger_,
                    "vstream_key = {};  failed to send face recognition event data to special group by callback ",
                    task_data.vstream_key, sg_group_callback_url);
              }
            }

        if (task_data.task_type == TASK_REGISTER_DESCRIPTOR)
        {
          if (best_register_index >= 0)
          {
            // everything is ok, register the descriptor
            cv::Rect r = enlargeFaceRect(face_data[best_register_index].face_rect, config.face_enlarge_scale);
            r = r & cv::Rect(0, 0, frame.cols, frame.rows);
            if (face_data[best_register_index].cosine_distance > 0.999)
              result.id_descriptor = face_data[best_register_index].id_descriptor;
            else
            {
              if (task_data.id_sgroup > 0)
                result.id_descriptor = addSGroupFaceDescriptor(task_data.id_sgroup, face_data[best_register_index].fd, frame(r));
              else
                result.id_descriptor = addFaceDescriptor(config.id_group, config.id_vstream, face_data[best_register_index].fd, frame(r));
            }

            if (result.id_descriptor > 0)
            {
              if (face_data[best_register_index].id_descriptor != result.id_descriptor)
              {
                result.comments = common_config.comments_new_descriptor;
                if (config.logs_level <= userver::logging::Level::kInfo)
                  USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
                    "vstream_key = {};  descriptor created: id = {}",
                    task_data.vstream_key, result.id_descriptor);
              } else
              {
                result.comments = common_config.comments_descriptor_exists;
                if (config.logs_level <= userver::logging::Level::kInfo)
                  USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
                    "vstream_key = {};  descriptor already exists: id = {}",
                    task_data.vstream_key, result.id_descriptor);
              }
              result.face_image = frame(r).clone();
              result.face_left = face_data[best_register_index].face_rect.x;
              result.face_top = face_data[best_register_index].face_rect.y;
              result.face_width = face_data[best_register_index].face_rect.width;
              result.face_height = face_data[best_register_index].face_rect.height;
            } else
              result.comments = common_config.comments_descriptor_creation_error;
          } else
          {
            if (!face_data.empty())
            {
              if (auto check_index = 0; !face_data[check_index].is_work_area)
                result.comments = common_config.comments_partial_face;
              else if (!face_data[check_index].is_frontal)
                result.comments = common_config.comments_non_frontal_face;
              else if (!face_data[check_index].is_non_blurry)
                result.comments = common_config.comments_blurry_face;
              else if (face_data[check_index].face_class_index != FACE_NORMAL)
                result.comments = common_config.comments_non_normal_face_class;
              else
                result.comments = common_config.comments_inference_error;
            } else
              result.comments = common_config.comments_no_faces;
          }
        }

        // drawing a frame and markers, saving a frame
        if (task_data.task_type == TASK_TEST)
        {
          for (auto& [face_rect, is_work_area, is_frontal, is_non_blurry, face_class_index, face_class_confidence, cosine_distance, fd, landmarks5, laplacian, ioa, id_descriptor, sg_descriptors] : face_data)
          {
            if (!landmarks5.empty())
              for (int k = 0; k < 5; ++k)
                cv::circle(frame, cv::Point(static_cast<int>(landmarks5.at<float>(k, 0)), static_cast<int>(landmarks5.at<float>(k, 1))), 1,
                  cv::Scalar(255 * (k * 2 > 2), 255 * (k * 2 > 0 && k * 2 < 8), 255 * (k * 2 < 6)), 4);
            cv::rectangle(frame, face_rect, cv::Scalar(0, 200, 0));
          }

          auto frame_indx = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
          AsyncNoSpan(fs_task_processor_,
            [&]
            {
              cv::imwrite(absl::Substitute("$0/frame_$1.jpg", std::filesystem::current_path().string(), frame_indx), frame);
            }).Get();
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
            "vstream_key = {};  task_type = {};  frame index: {}",
            task_data.vstream_key, static_cast<int>(task_data.task_type), frame_indx);
        }
      } else
      {
        if (task_data.task_type == TASK_REGISTER_DESCRIPTOR)
        {
          result.id_descriptor = 0;
          result.comments = common_config.comments_inference_error;
        }
      }
    } catch (const std::exception& e)
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "vstream_key = {};  {}",
          task_data.vstream_key, e.what());
      if (task_data.task_type == TASK_RECOGNIZE)
      {
        if (config.delay_after_error.count() > 0)
        {
          if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
              "vstream_key = {};  delay for {}ms",
              task_data.vstream_key, config.delay_after_error.count());
          delay_between_frames = config.delay_after_error;
        } else
        {
          stopWorkflow(std::move(task_data.vstream_key));
          return {.comments = "Error during pipeline execution"};
        }
      } else
        return {.comments = "Error during pipeline execution"};
    }

    if (config.logs_level <= userver::logging::Level::kDebug || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "End processPipeline: vstream_key = {};",
        task_data.vstream_key);

    if (task_data.task_type == TASK_RECOGNIZE)
      nextPipeline(std::move(task_data), delay_between_frames);

    return result;
  }

  void Workflow::loadDNNStatsData()
  {
    auto f_name = std::filesystem::current_path().string() + "/dnn_stats_data.json";
    if (!std::filesystem::exists(f_name))
      return;

    try
    {
      std::fstream f(f_name);
      auto json_data = userver::formats::json::FromStream(f);
      auto dnn_stats_ptr = dnn_stats_data.Lock();
      dnn_stats_ptr->clear();
      for (const auto& item : json_data["data"])
        (*dnn_stats_ptr)[item["id_vstream"].As<int32_t>()] = DNNStatsData{
          item["fd_count"].As<int32_t>(),
          item["fc_count"].As<int32_t>(),
          item["fr_count"].As<int32_t>()};
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(logger_) << e.what();
    }
  }

  void Workflow::saveDNNStatsData() const
  {
    userver::formats::json::ValueBuilder list_data;
    DNNStatsData all_data;
    // scope to access concurrent variable
    {
      auto dnn_stats_ptr = dnn_stats_data.Lock();
      for (const auto& [fst, snd] : *dnn_stats_ptr)
      {
        all_data.fd_count += snd.fd_count;
        all_data.fc_count += snd.fc_count;
        all_data.fr_count += snd.fr_count;
        userver::formats::json::ValueBuilder v;
        v["id_vstream"] = fst;
        v["fd_count"] = snd.fd_count;
        v["fc_count"] = snd.fc_count;
        v["fr_count"] = snd.fr_count;
        list_data.PushBack(std::move(v));
      }
    }
    userver::formats::json::ValueBuilder json_data;
    json_data["all"]["fd_count"] = all_data.fd_count;
    json_data["all"]["fc_count"] = all_data.fc_count;
    json_data["all"]["fr_count"] = all_data.fr_count;
    json_data["data"] = std::move(list_data);
    try
    {
      std::ofstream f(std::filesystem::current_path().string() + "/dnn_stats_data.json");
      f << ToString(json_data.ExtractValue());
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(logger_) << e.what();
    }
  }

  void Workflow::doOldLogMaintenance() const
  {
    LOG_INFO_TO(logger_, "Removing obsolete entries from the log_faces table");

    auto tp = std::chrono::system_clock::now() - local_config_.log_faces_ttl;
    try
    {
      pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
        SQL_REMOVE_OLD_LOG_FACES, userver::storages::postgres::TimePointTz{tp});
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(logger_) << e.what();
      return;
    }

    LOG_INFO_TO(logger_, "Removing outdated screenshots");
    const HashSet<std::string> img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tiff", ".dat", ".json"};
    if (std::filesystem::exists(local_config_.screenshots_path))
      for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(local_config_.screenshots_path))
        if (dir_entry.is_regular_file() && img_extensions.contains(dir_entry.path().extension().string()))
        {
          if (auto t = std::chrono::file_clock::to_sys(dir_entry.last_write_time()); t < tp)
          {
            std::error_code ec;
            std::filesystem::remove(dir_entry, ec);
            if (ec)
              LOG_ERROR_TO(logger_) << ec.message();
          }
        }
  }

  void Workflow::doFlagDeletedMaintenance() const
  {
    LOG_DEBUG_TO(logger_, "Deleting marked records from the database");

    auto tp = std::chrono::system_clock::now() - local_config_.flag_deleted_ttl;
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      trx.Execute(SQL_DELETE_VIDEO_STREAMS_MARKED, userver::storages::postgres::TimePointTz{tp});
      trx.Execute(SQL_DELETE_FACE_DESCRIPTORS_MARKED, userver::storages::postgres::TimePointTz{tp});
      trx.Execute(SQL_DELETE_LINK_DESCRIPTOR_VSTREAM_MARKED, userver::storages::postgres::TimePointTz{tp});
      trx.Execute(SQL_DELETE_SPECIAL_GROUPS_MARKED, userver::storages::postgres::TimePointTz{tp});
      trx.Execute(SQL_DELETE_LINK_DESCRIPTOR_SGROUP_MARKED, userver::storages::postgres::TimePointTz{tp});
      trx.Commit();
    } catch (const std::exception& e)
    {
      trx.Rollback();
      LOG_ERROR_TO(logger_) << e.what();
    }
  }

  void Workflow::doCopyEventsMaintenance() const
  {
    LOG_DEBUG_TO(logger_, "Copying event data");
    try
    {
      for (auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_GET_LOG_COPY_DATA); const auto& row : result)
      {
        auto id_log = row[DatabaseFields::ID_LOG].As<int64_t>();
        auto id_group = row[DatabaseFields::ID_GROUP].As<int32_t>();
        auto s_uuid = absl::StrReplaceAll(row[DatabaseFields::LOG_UUID].As<std::string>(), {{"-", ""}});
        auto ext_event_uuid = row[DatabaseFields::EXT_EVENT_UUID].As<std::string>();
        auto log_date = row[DatabaseFields::LOG_DATE].As<userver::storages::postgres::TimePointTz>();
        auto path_suffix = absl::Substitute("group_$0/$1/$2/$3/$4/", id_group, s_uuid[0], s_uuid[1], s_uuid[2], s_uuid[3]);
        auto orig_path_prefix = absl::StrCat(local_config_.screenshots_path, path_suffix);
        auto orig_path_json = absl::StrCat(orig_path_prefix, s_uuid, JSON_SUFFIX);
        auto orig_path_dat = absl::StrCat(orig_path_prefix, s_uuid, DATA_FILE_SUFFIX);
        std::error_code ec;
        if (!std::filesystem::exists(orig_path_json, ec))
        {
          // trying the shorter path for compatibility with the old project
          path_suffix = absl::Substitute("group_$0/$1/$2/$3/", id_group, s_uuid[0], s_uuid[1], s_uuid[2]);
          orig_path_prefix = absl::StrCat(local_config_.screenshots_path, path_suffix);
          orig_path_json = absl::StrCat(orig_path_prefix, s_uuid, JSON_SUFFIX);
          orig_path_dat = absl::StrCat(orig_path_prefix, s_uuid, DATA_FILE_SUFFIX);
        }
        if (std::filesystem::exists(orig_path_json, ec))
        {
          auto copy_path_prefix = absl::StrCat(local_config_.events_path, path_suffix);
          auto copy_path_json = absl::StrCat(copy_path_prefix, s_uuid, JSON_SUFFIX);
          std::filesystem::create_directories(copy_path_prefix, ec);
          if (ec)
          {
            LOG_ERROR_TO(logger_) << ec.message();
            continue;
          }

          // copy the JSON data adding the external event uuid
          std::fstream fsr(orig_path_json);
          auto json_data = userver::formats::json::ValueBuilder(userver::formats::json::FromStream(fsr));
          json_data["event_uuid"] = ext_event_uuid;
          std::ofstream fsw(copy_path_json);
          fsw << ToString(json_data.ExtractValue());

          // append face descriptor data
          if (const auto f_size = std::filesystem::file_size(orig_path_dat, ec); !ec && f_size > 0)
          {
            auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
            try
            {
              std::ifstream fr_data(orig_path_dat, std::ios::in | std::ios::binary);
              std::string s_data(f_size, '\0');
              fr_data.read(s_data.data(), static_cast<std::streamsize>(f_size));

              auto time = absl::FromChrono(log_date.GetUnderlying());
              auto group_part = absl::StrCat("group_", id_group, "/");
              auto events_file = absl::StrCat(local_config_.events_path, group_part,
                absl::FormatTime(DATE_FORMAT, time, absl::LocalTimeZone()),
                ".dat");
              std::ofstream fw_data(events_file, std::ios::app);
              fw_data.write(s_data.data(), static_cast<std::streamsize>(f_size));

              trx.Execute(SQL_UPDATE_LOG_COPY_DATA, id_log);
              trx.Commit();
            } catch (const std::exception& e)
            {
              trx.Rollback();
              LOG_ERROR_TO(logger_) << e.what();
            }
          }
        }
      }
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(logger_) << e.what();
    }
  }

  void Workflow::doOldEventsMaintenance() const
  {
    LOG_INFO_TO(logger_, "Removing outdated events");

    const auto tp = std::chrono::system_clock::now() - local_config_.events_ttl;
    const HashSet<std::string> img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tiff", ".dat", ".json"};
    if (std::filesystem::exists(local_config_.events_path))
      for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(local_config_.events_path))
        if (dir_entry.is_regular_file() && img_extensions.contains(dir_entry.path().extension().string()))
        {
          if (auto t = std::chrono::file_clock::to_sys(dir_entry.last_write_time()); t < tp)
          {
            std::error_code ec;
            std::filesystem::remove(dir_entry, ec);
            if (ec)
              LOG_ERROR_TO(logger_) << ec.message();
          }
        }
  }

  void Workflow::nextPipeline(TaskData&& task_data, const std::chrono::milliseconds delay)
  {
    userver::engine::InterruptibleSleepFor(delay);

    bool do_next = false;
    bool is_timeout = false;

    // scope for accessing concurrent variable
    {
      const auto now = std::chrono::steady_clock::now();
      auto data_ptr = vstream_timeouts.Lock();
      if (data_ptr->contains(task_data.vstream_key))
        if (data_ptr->at(task_data.vstream_key) < now)
        {
          data_ptr->erase(task_data.vstream_key);
          is_timeout = true;
        }
    }

    // scope for accessing concurrent variable
    {
      auto data_ptr = being_processed_vstreams.Lock();
      if (data_ptr->contains(task_data.vstream_key))
      {
        if (data_ptr->at(task_data.vstream_key) && !is_timeout)
          do_next = true;
        else
          data_ptr->erase(task_data.vstream_key);
      }
    }

    if (is_timeout)
      LOG_INFO_TO(logger_,
        "Stopping a workflow by timeout: vstream_key = {};",
        task_data.vstream_key);

    if (do_next)
      tasks_.Detach(AsyncNoSpan(task_processor_, &Workflow::processPipeline, this, std::move(task_data)));
  }

  // Inference pipeline functions
  cv::Mat Workflow::preprocessImage(const cv::Mat& img, const int width, const int height, float& scale)
  {
    int w, h;
    const auto r_w = width / (img.cols * 1.0);
    if (const auto r_h = height / (img.rows * 1.0); r_h > r_w)
    {
      w = width;
      h = static_cast<int>(r_w * img.rows);
    } else
    {
      w = static_cast<int>(r_h * img.cols);
      h = height;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    scale = static_cast<float>(h) / static_cast<float>(img.rows);

    return out;
  }

  bool Workflow::detectFaces(const TaskData& task_data, const cv::Mat& frame, const VStreamConfig& config, std::vector<FaceDetection>& detected_faces)
  {
    decltype(CommonConfig::dnn_fd_model_name) dnn_fd_model_name = "scrfd";
    decltype(CommonConfig::dnn_fd_input_width) dnn_fd_input_width = 320;
    decltype(CommonConfig::dnn_fd_input_height) dnn_fd_input_height = 320;
    decltype(CommonConfig::dnn_fd_input_tensor_name) dnn_fd_input_tensor_name = "input.1";

    // scope for accessing cache
    {
      if (auto cache = common_config_cache_.Get(); cache->getCommonConfig().contains(config.id_group))
      {
        dnn_fd_model_name = cache->getCommonConfig().at(config.id_group).dnn_fd_model_name;
        dnn_fd_input_width = cache->getCommonConfig().at(config.id_group).dnn_fd_input_width;
        dnn_fd_input_height = cache->getCommonConfig().at(config.id_group).dnn_fd_input_height;
        dnn_fd_input_tensor_name = cache->getCommonConfig().at(config.id_group).dnn_fd_input_tensor_name;
      }
    }

    std::unique_ptr<tc::InferenceServerHttpClient> triton_client;
    auto err = tc::InferenceServerHttpClient::Create(&triton_client, config.dnn_fd_inference_server, false);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create inference client: {}",
          err.Message());
      return false;
    }

    float scale = 1.0f;

    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  before image preprocessing for face detection",
        task_data.vstream_key);
    cv::Mat pr_img = preprocessImage(frame, dnn_fd_input_width, dnn_fd_input_height, scale);
    int channels = 3;
    int input_size = channels * dnn_fd_input_width * dnn_fd_input_height;
    std::vector<float> input_buffer(input_size);
    for (int c = 0; c < channels; ++c)
      for (int h = 0; h < dnn_fd_input_height; ++h)
        for (int w = 0; w < dnn_fd_input_width; ++w)
          input_buffer[c * dnn_fd_input_height * dnn_fd_input_width + h * dnn_fd_input_width + w] = (static_cast<float>(pr_img.at<cv::Vec3b>(h, w)[2 - c]) - 127.5f) / 128.0f;
    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  after image preprocessing for face detection",
        task_data.vstream_key);

    std::vector<uint8_t> input_data(input_size * sizeof(float));
    memcpy(input_data.data(), input_buffer.data(), input_data.size());
    std::vector<int64_t> shape = {1, channels, dnn_fd_input_height, dnn_fd_input_width};
    tc::InferInput* input;
    err = tc::InferInput::Create(&input, dnn_fd_input_tensor_name, shape, "FP32");
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create input data: {}", err.Message());
      return false;
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);
    std::vector inputs = {input_ptr.get()};
    err = input_ptr->AppendRaw(input_data);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to append input data: {}",
          err.Message());
      return false;
    }

    std::vector output_tensors = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};
    std::vector<const tc::InferRequestedOutput*> outputs;
    outputs.reserve(output_tensors.size());
    std::vector<std::shared_ptr<tc::InferRequestedOutput>> outputs_ptr;
    outputs_ptr.reserve(output_tensors.size());
    for (size_t i = 0; i < output_tensors.size(); ++i)
    {
      tc::InferRequestedOutput* p;
      err = tc::InferRequestedOutput::Create(&p, output_tensors[i]);
      if (!err.IsOk())
      {
        if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
            "Error! Unable to create output data: {}",
            err.Message());
        return false;
      }
      outputs_ptr.emplace_back(p);
      outputs.emplace_back(outputs_ptr[i].get());
    }

    tc::InferOptions options(dnn_fd_model_name);
    options.model_version_ = "";
    tc::InferResult* result;

    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  before inference face detection",
        task_data.vstream_key);
    userver::engine::AsyncNoSpan(fs_task_processor_,
      [&]
      {
        err = triton_client->Infer(&result, options, inputs, outputs);
      }).Get();
    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  after inference face detection",
        task_data.vstream_key);

    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to send inference request: {}",
          err.Message());
      return false;
    }

    std::shared_ptr<tc::InferResult> result_ptr(result);
    if (!result_ptr->RequestStatus().IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to receive inference result: {}",
          err.Message());
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kDebug || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "vstream_key = {};  inference face detection OK",
        task_data.vstream_key);

    std::vector feat_stride = {8, 16, 32};

    detected_faces.clear();
    for (size_t i = 0; i < feat_stride.size(); ++i)
    {
      constexpr int fmc = 3;
      const float* scores_data;
      size_t scores_size;
      result_ptr->RawData(output_tensors[i], reinterpret_cast<const uint8_t**>(&scores_data), &scores_size);

      const float* bbox_preds_data;
      size_t bbox_preds_size;
      result_ptr->RawData(output_tensors[i + fmc], reinterpret_cast<const uint8_t**>(&bbox_preds_data), &bbox_preds_size);
      auto bbox_preds = cv::Mat(static_cast<int>(bbox_preds_size / 4 / sizeof(float)), 4, CV_32F, const_cast<float*>(bbox_preds_data));
      bbox_preds *= feat_stride[i];

      const float* kps_preds_data;
      size_t kps_preds_size;
      result_ptr->RawData(output_tensors[i + fmc * 2], reinterpret_cast<const uint8_t**>(&kps_preds_data), &kps_preds_size);
      auto kps_preds = cv::Mat(static_cast<int>(kps_preds_size / 10 / sizeof(float)), 10, CV_32F, const_cast<float*>(kps_preds_data));
      kps_preds *= feat_stride[i];

      int height = dnn_fd_input_height / feat_stride[i];
      int width = dnn_fd_input_width / feat_stride[i];
      for (int k = 0; k < height * width; ++k)
      {
        auto px = static_cast<float>(feat_stride[i] * (k % height));
        auto py = static_cast<float>(feat_stride[i] * static_cast<int>(k / height));
        if (scores_data[2 * k] >= config.face_confidence)
        {
          FaceDetection det{};
          det.face_confidence = scores_data[2 * k];
          det.bbox[0] = (px - bbox_preds.at<float>(2 * k, 0)) / scale;
          det.bbox[1] = (py - bbox_preds.at<float>(2 * k, 1)) / scale;
          det.bbox[2] = (px + bbox_preds.at<float>(2 * k, 2)) / scale;
          det.bbox[3] = (py + bbox_preds.at<float>(2 * k, 3)) / scale;
          for (int j = 0; j < 5; ++j)
          {
            det.landmark[2 * j] = (px + kps_preds.at<float>(2 * k, 2 * j)) / scale;
            det.landmark[2 * j + 1] = (py + kps_preds.at<float>(2 * k, 2 * j + 1)) / scale;
          }
          detected_faces.emplace_back(det);
        }
        if (scores_data[2 * k + 1] >= config.face_confidence)
        {
          FaceDetection det{};
          det.face_confidence = scores_data[2 * k + 1];
          det.bbox[0] = (px - bbox_preds.at<float>(2 * k + 1, 0)) / scale;
          det.bbox[1] = (py - bbox_preds.at<float>(2 * k + 1, 1)) / scale;
          det.bbox[2] = (px + bbox_preds.at<float>(2 * k + 1, 2)) / scale;
          det.bbox[3] = (py + bbox_preds.at<float>(2 * k + 1, 3)) / scale;
          for (int j = 0; j < 5; ++j)
          {
            det.landmark[2 * j] = (px + kps_preds.at<float>(2 * k + 1, 2 * j)) / scale;
            det.landmark[2 * j + 1] = (py + kps_preds.at<float>(2 * k + 1, 2 * j + 1)) / scale;
          }
          detected_faces.emplace_back(det);
        }
      }
    }

    nms(detected_faces);

    return true;
  }

  bool Workflow::inferFaceClass(const TaskData& task_data, const cv::Mat& aligned_face, const VStreamConfig& config, std::vector<FaceClass>& face_classes)
  {
    decltype(CommonConfig::dnn_fc_model_name) dnn_fc_model_name = "genet";
    decltype(CommonConfig::dnn_fc_input_width) dnn_fc_input_width = 192;
    decltype(CommonConfig::dnn_fc_input_height) dnn_fc_input_height = 192;
    decltype(CommonConfig::dnn_fc_input_tensor_name) dnn_fc_input_tensor_name = "input.1";
    decltype(CommonConfig::dnn_fc_output_tensor_name) dnn_fc_output_tensor_name = "419";
    decltype(CommonConfig::dnn_fc_output_size) dnn_fc_output_size = 3;

    // scope for accessing cache
    {
      if (auto cache = common_config_cache_.Get(); cache->getCommonConfig().contains(config.id_group))
      {
        dnn_fc_model_name = cache->getCommonConfig().at(config.id_group).dnn_fc_model_name;
        dnn_fc_input_width = cache->getCommonConfig().at(config.id_group).dnn_fc_input_width;
        dnn_fc_input_height = cache->getCommonConfig().at(config.id_group).dnn_fc_input_height;
        dnn_fc_input_tensor_name = cache->getCommonConfig().at(config.id_group).dnn_fc_input_tensor_name;
        dnn_fc_output_tensor_name = cache->getCommonConfig().at(config.id_group).dnn_fc_output_tensor_name;
        dnn_fc_output_size = cache->getCommonConfig().at(config.id_group).dnn_fc_output_size;
      }
    }

    std::unique_ptr<tc::InferenceServerHttpClient> triton_client;
    auto err = tc::InferenceServerHttpClient::Create(&triton_client, config.dnn_fc_inference_server, false);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create inference client: {}",
          err.Message());
      return false;
    }

    int channels = 3;
    int input_size = channels * dnn_fc_input_width * dnn_fc_input_height;
    std::vector<float> input_buffer(input_size);

    for (int c = 0; c < channels; ++c)
      for (int h = 0; h < dnn_fc_input_height; ++h)
        for (int w = 0; w < dnn_fc_input_width; ++w)
        {
          float mean = 0.0f;
          float std_d = 1.0f;
          if (c == 0)
          {
            mean = 0.485;
            std_d = 0.229;
          }
          if (c == 1)
          {
            mean = 0.456;
            std_d = 0.224;
          }
          if (c == 2)
          {
            mean = 0.406;
            std_d = 0.225;
          }
          input_buffer[c * dnn_fc_input_height * dnn_fc_input_width + h * dnn_fc_input_width + w] =
            (static_cast<float>(aligned_face.at<cv::Vec3b>(h, w)[2 - c]) / 255.0f - mean) / std_d;
        }
    std::vector<uint8_t> input_data(input_size * sizeof(float));
    memcpy(input_data.data(), input_buffer.data(), input_data.size());
    std::vector<int64_t> shape = {1, channels, dnn_fc_input_height, dnn_fc_input_width};
    tc::InferInput* input;
    err = tc::InferInput::Create(&input, dnn_fc_input_tensor_name, shape, "FP32");
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create input data: {}",
          err.Message());
      return false;
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);
    err = input_ptr->AppendRaw(input_data);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to append input data: {}",
          err.Message());
      return false;
    }
    std::vector inputs = {input_ptr.get()};

    tc::InferRequestedOutput* output;
    err = tc::InferRequestedOutput::Create(&output, dnn_fc_output_tensor_name);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create output data: {}",
          err.Message());
      return false;
    }
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};

    tc::InferOptions options(dnn_fc_model_name);
    options.model_version_ = "";
    tc::InferResult* result;

    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  before inference face class",
        task_data.vstream_key);
    userver::engine::AsyncNoSpan(fs_task_processor_,
      [&]
      {
        err = triton_client->Infer(&result, options, inputs, outputs);
      }).Get();
    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  after inference face class",
        task_data.vstream_key);

    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to send inference request: {}",
          err.Message());
      return false;
    }

    std::shared_ptr<tc::InferResult> result_ptr(result);
    if (!result_ptr->RequestStatus().IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError, "Error! Unable to receive inference result");
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kDebug || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "vstream_key = {};  inference face class OK",
        task_data.vstream_key);

    const float* result_data;
    size_t output_size;
    err = result_ptr->RawData(dnn_fc_output_tensor_name, reinterpret_cast<const uint8_t**>(&result_data), &output_size);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Failed to get output: {}",
          err.Message());
      return false;
    }

    std::vector<float> scores;
    scores.assign(result_data, result_data + dnn_fc_output_size);
    face_classes = softMax(scores);

    return true;
  }

  bool Workflow::extractFaceDescriptor(const TaskData& task_data, const cv::Mat& aligned_face, const VStreamConfig& config, FaceDescriptor& face_descriptor)
  {
    decltype(CommonConfig::dnn_fr_model_name) dnn_fr_model_name = "arcface";
    decltype(CommonConfig::dnn_fr_input_width) dnn_fr_input_width = 112;
    decltype(CommonConfig::dnn_fr_input_height) dnn_fr_input_height = 112;
    decltype(CommonConfig::dnn_fr_input_tensor_name) dnn_fr_input_tensor_name = "input.1";
    decltype(CommonConfig::dnn_fr_output_tensor_name) dnn_fr_output_tensor_name = "683";
    decltype(CommonConfig::dnn_fr_output_size) dnn_fr_output_size = 512;

    // scope for accessing cache
    {
      if (auto cache = common_config_cache_.Get(); cache->getCommonConfig().contains(config.id_group))
      {
        dnn_fr_model_name = cache->getCommonConfig().at(config.id_group).dnn_fr_model_name;
        dnn_fr_input_width = cache->getCommonConfig().at(config.id_group).dnn_fr_input_width;
        dnn_fr_input_height = cache->getCommonConfig().at(config.id_group).dnn_fr_input_height;
        dnn_fr_input_tensor_name = cache->getCommonConfig().at(config.id_group).dnn_fr_input_tensor_name;
        dnn_fr_output_tensor_name = cache->getCommonConfig().at(config.id_group).dnn_fr_output_tensor_name;
        dnn_fr_output_size = cache->getCommonConfig().at(config.id_group).dnn_fr_output_size;
      }
    }

    std::unique_ptr<tc::InferenceServerHttpClient> triton_client;
    auto err = tc::InferenceServerHttpClient::Create(&triton_client, config.dnn_fr_inference_server, false);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create inference client: {}",
          err.Message());
      return false;
    }

    int channels = 3;
    int input_size = channels * dnn_fr_input_width * dnn_fr_input_height;
    std::vector<float> input_buffer(input_size);
    for (int c = 0; c < channels; ++c)
      for (int h = 0; h < dnn_fr_input_height; ++h)
        for (int w = 0; w < dnn_fr_input_width; ++w)
          if (dnn_fr_model_name == "arcface")
            input_buffer[c * dnn_fr_input_height * dnn_fr_input_width + h * dnn_fr_input_width + w] = static_cast<float>(aligned_face.at<cv::Vec3b>(h, w)[2 - c]) / 127.5f - 1.0f;
          else
            input_buffer[c * dnn_fr_input_height * dnn_fr_input_width + h * dnn_fr_input_width + w] = (static_cast<float>(aligned_face.at<cv::Vec3b>(h, w)[2 - c]) - 127.5f) / 128.0f;
    std::vector<uint8_t> input_data(input_size * sizeof(float));
    memcpy(input_data.data(), input_buffer.data(), input_data.size());
    std::vector<int64_t> shape = {1, channels, dnn_fr_input_height, dnn_fr_input_width};
    tc::InferInput* input;
    err = tc::InferInput::Create(&input, dnn_fr_input_tensor_name, shape, "FP32");
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create input data: {}",
          err.Message());
      return false;
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);
    err = input_ptr->AppendRaw(input_data);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to append input data: {}",
          err.Message());
      return false;
    }
    std::vector inputs = {input_ptr.get()};

    tc::InferRequestedOutput* output;
    err = tc::InferRequestedOutput::Create(&output, dnn_fr_output_tensor_name);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to create output data: {}",
          err.Message());
      return false;
    }
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};

    tc::InferOptions options(dnn_fr_model_name);
    options.model_version_ = "";
    tc::InferResult* result;

    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  before inference for extracting descriptor",
        task_data.vstream_key);
    userver::engine::AsyncNoSpan(fs_task_processor_,
      [&]
      {
        err = triton_client->Infer(&result, options, inputs, outputs);
      }).Get();
    if (config.logs_level <= userver::logging::Level::kTrace || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {};  after inference for extracting descriptor",
        task_data.vstream_key);

    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to send inference request: {}",
          err.Message());
      return false;
    }

    std::shared_ptr<tc::InferResult> result_ptr(result);
    if (!result_ptr->RequestStatus().IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Unable to receive inference result: {}",
          err.Message());
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kDebug || task_data.task_type == TASK_TEST)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "vstream_key = {};  inference face descriptor extractor OK",
        task_data.vstream_key);

    const float* result_data;
    size_t output_size;
    err = result_ptr->RawData(dnn_fr_output_tensor_name, reinterpret_cast<const uint8_t**>(&result_data), &output_size);
    if (!err.IsOk())
    {
      if (config.logs_level <= userver::logging::Level::kError || task_data.task_type == TASK_TEST)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "Error! Failed to get output: {}",
          err.Message());
      return false;
    }

    auto m_result = cv::Mat(1, dnn_fr_output_size, CV_32F, const_cast<float*>(result_data));
    face_descriptor = m_result.clone();

    return true;
  }

  int64_t Workflow::addLogFace(const int32_t id_vstream, const userver::storages::postgres::TimePointTz& log_date,
    int32_t id_descriptor, const double quality, const cv::Rect& face_rect, const std::string& screenshot_url, const boost::uuids::uuid& uuid,
    const CopyEventData copy_event_data) const
  {
    const userver::storages::postgres::Query query{SQL_ADD_LOG_FACE};
    int64_t result = -1;
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      const auto res = trx.Execute(query,
        id_vstream, log_date, id_descriptor > 0 ? std::optional(id_descriptor) : std::nullopt, quality, face_rect.x, face_rect.y, face_rect.width, face_rect.height, screenshot_url, uuid, static_cast<int32_t>(copy_event_data));
      if (!res.IsEmpty())
        result = res.AsSingleRow<int64_t>();
      trx.Commit();
    } catch (const std::exception& e)
    {
      trx.Rollback();
      LOG_ERROR_TO(logger_) << e.what();
    }

    return result;
  }

  int32_t Workflow::addFaceDescriptor(const int32_t id_group, const int32_t id_vstream, const FaceDescriptor& fd, const cv::Mat& f_img,
    const int32_t id_parent)
  {
    decltype(CommonConfig::dnn_fr_output_size) dnn_fr_output_size = 512;
    // scope for accessing cache
    {
      if (const auto cache = common_config_cache_.Get(); cache->getCommonConfig().contains(id_group))
        dnn_fr_output_size = cache->getCommonConfig().at(id_group).dnn_fr_output_size;
    }

    int32_t id_descriptor = -1;
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      std::vector fd_data(fd.data, fd.data + dnn_fr_output_size * sizeof(float));
      const auto res = trx.Execute(SQL_ADD_FACE_DESCRIPTOR, id_group, userver::storages::postgres::Bytea(fd_data),
        id_parent > 0 ? std::optional(id_parent) : std::nullopt);
      id_descriptor = res.AsSingleRow<int32_t>();

      std::vector<uchar> buff;
      cv::imencode(".jpg", f_img, buff);
      trx.Execute(SQL_ADD_DESCRIPTOR_IMAGE, id_descriptor, MIME_IMAGE, userver::storages::postgres::Bytea(buff));
      if (id_parent == 0)
        trx.Execute(Api::SQL_ADD_LINK_DESCRIPTOR_VSTREAM, id_group, id_vstream, id_descriptor);
      trx.Commit();
    } catch (const std::exception& e)
    {
      trx.Rollback();
      id_descriptor = -1;
      LOG_ERROR_TO(logger_) << e.what();
    }

    return id_descriptor;
  }

  int32_t Workflow::addSGroupFaceDescriptor(const int32_t id_sgroup, const FaceDescriptor& fd, const cv::Mat& f_img)
  {
    int32_t id_descriptor = -1;
    int32_t id_group = -1;
    // scope for accessing cache
    {
      if (const auto sg_config = sg_config_cache_.Get(); sg_config->getMap().contains(id_sgroup))
        id_group = sg_config->getData().at(sg_config->getMap().at(id_sgroup)).id_group;
    }
    if (id_group <= 0)
      return id_descriptor;

    decltype(CommonConfig::dnn_fr_output_size) dnn_fr_output_size = 512;
    // scope for accessing cache
    {
      if (const auto cache = common_config_cache_.Get(); cache->getCommonConfig().contains(id_group))
        dnn_fr_output_size = cache->getCommonConfig().at(id_group).dnn_fr_output_size;
    }

    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      std::vector fd_data(fd.data, fd.data + dnn_fr_output_size * sizeof(float));
      const auto res = trx.Execute(SQL_ADD_FACE_DESCRIPTOR, id_group, userver::storages::postgres::Bytea(fd_data), std::optional<int32_t>(std::nullopt));
      id_descriptor = res.AsSingleRow<int32_t>();

      std::vector<uchar> buff;
      cv::imencode(".jpg", f_img, buff);
      trx.Execute(SQL_ADD_DESCRIPTOR_IMAGE, id_descriptor, MIME_IMAGE, userver::storages::postgres::Bytea(buff));
      trx.Execute(SQL_ADD_LINK_DESCRIPTOR_SGROUP, id_sgroup, id_descriptor);
      trx.Commit();
    } catch (const std::exception& e)
    {
      trx.Rollback();
      id_descriptor = -1;
      LOG_ERROR_TO(logger_) << e.what();
    }

    return id_descriptor;
  }
}  // namespace Frs
