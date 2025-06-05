#include <cmath>
#include <filesystem>

#include <absl/strings/str_format.h>
#include <absl/strings/substitute.h>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <http_client.h>
#include <userver/engine/sleep.hpp>
#include <userver/engine/wait_all_checked.hpp>
#include <userver/fs/write.hpp>
#include <userver/http/common_headers.hpp>
#include <userver/http/content_type.hpp>
#include <userver/yaml_config/merge_schemas.hpp>

#include "lprs_api.hpp"
#include "lprs_workflow.hpp"

namespace tc = triton::client;

namespace Lprs
{
  inline bool cmp_vehicles(const Vehicle& a, const Vehicle& b)
  {
    return a.confidence > b.confidence;
  }

  inline bool cmp_plates(const LicensePlate& a, const LicensePlate& b)
  {
    return a.confidence > b.confidence;
  }

  inline bool hasIntersection(float lbox[4], float rbox[4])
  {
    const float inter_box[] =
      {
        std::max(lbox[0], rbox[0]),  // left
        std::min(lbox[2], rbox[2]),  // right
        std::max(lbox[1], rbox[1]),  // top
        std::min(lbox[3], rbox[3]),  // bottom
      };

    return inter_box[0] < inter_box[1] && inter_box[2] < inter_box[3];
  }

  inline float iou(const cv::Rect2f& r1, const cv::Rect2f& r2)
  {
    const auto r_intersection = r1 & r2;
    return r_intersection.area() / (r1.area() + r2.area() - r_intersection.area());
  }

  // non-maximum suppression algorithm for vehicle detection
  inline void nms_vehicles(std::vector<Vehicle>& vehicles, const float threshold)
  {
    std::ranges::sort(vehicles, cmp_vehicles);
    for (size_t m = 0; m < vehicles.size(); ++m)
    {
      auto& [bbox, confidence, is_special, license_plates] = vehicles[m];
      for (size_t n = m + 1; n < vehicles.size(); ++n)
      {
        auto r1 = cv::Rect2f(cv::Point2f{bbox[0], bbox[1]}, cv::Point2f{bbox[2], bbox[3]});
        if (auto r2 = cv::Rect2f(cv::Point2f{vehicles[n].bbox[0], vehicles[n].bbox[1]}, cv::Point2f{vehicles[n].bbox[2], vehicles[n].bbox[3]}); iou(r1, r2) > threshold)
        {
          vehicles.erase(vehicles.begin() + static_cast<int>(n));
          --n;
        }
      }
    }
  }

  // non-maximum suppression algorithm for plate detection
  inline void nms_plates(std::vector<LicensePlate>& dets)
  {
    std::ranges::sort(dets, cmp_plates);
    for (size_t m = 0; m < dets.size(); ++m)
    {
      auto& [bbox, confidence, kpts, plate_class, plate_numbers] = dets[m];
      for (size_t n = m + 1; n < dets.size(); ++n)
      {
        if (plate_class == dets[n].plate_class && hasIntersection(bbox, dets[n].bbox))
        {
          dets.erase(dets.begin() + static_cast<int>(n));
          --n;
        }
      }
    }
  }

  inline bool cmp_chars_conf(const CharData& a, const CharData& b)
  {
    return a.confidence > b.confidence;
  }

  inline bool cmp_chars_position(const CharData& a, const CharData& b)
  {
    constexpr int32_t xmin = 0;

    // single line license plate number
    if (a.plate_class == PLATE_CLASS_RU_1)
      return a.bbox[xmin] < b.bbox[xmin];

    // double line license plate number
    if (a.plate_class == PLATE_CLASS_RU_1A)
    {
      constexpr int32_t ymin = 1;
      constexpr int32_t ymax = 3;
      if (a.bbox[ymax] < b.bbox[ymin])
        return true;

      if (a.bbox[ymin] > b.bbox[ymax])
        return false;

      const auto y = a.bbox[ymin] + 0.5 * (a.bbox[ymax] - a.bbox[ymin]);
      if (y < b.bbox[ymin])
        return true;

      if (y > b.bbox[ymax])
        return false;
    }

    return a.bbox[xmin] < b.bbox[xmin];
  }

  // non-maximum suppression algorithm for char recognition
  inline void nms_chars(std::vector<CharData>& chars, const float threshold)
  {
    std::ranges::sort(chars, cmp_chars_conf);
    for (size_t m = 0; m < chars.size(); ++m)
    {
      auto& [bbox, confidence, char_class, plate_class] = chars[m];
      for (size_t n = m + 1; n < chars.size(); ++n)
      {
        auto r1 = cv::Rect2f(cv::Point2f{bbox[0], bbox[1]}, cv::Point2f{bbox[2], bbox[3]});
        if (auto r2 = cv::Rect2f(cv::Point2f{chars[n].bbox[0], chars[n].bbox[1]}, cv::Point2f{chars[n].bbox[2], chars[n].bbox[3]}); char_class == chars[n].char_class && iou(r1, r2) > threshold)
        {
          chars.erase(chars.begin() + static_cast<int>(n));
          --n;
        }
      }
    }
  }

  inline float euclidean_distance(const float x1, const float y1, const float x2, const float y2)
  {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  }

  inline std::vector<float> softMax(const std::vector<float>& v)
  {
    if (v.empty())
      return {};

    std::vector<float> r(v.size());
    float s = 0.0f;
    for (const float i : v)
      s += exp(i);
    for (size_t i = 0; i < v.size(); ++i)
      r[i] = exp(v[i]) / s;

    return r;
  }

  inline std::vector<std::vector<cv::Point>> convertToAbsolute(const std::vector<std::vector<cv::Point2f>>& work_area, int32_t width, int32_t height)
  {
    std::vector<std::vector<cv::Point>> wa(work_area.size());
    for (size_t i = 0; i < work_area.size(); ++i)
    {
      wa[i].reserve(work_area[i].size());
      for (size_t j = 0; j < work_area[i].size(); ++j)
        wa[i].emplace_back(static_cast<int>(work_area[i][j].x * width / 100.0f), static_cast<int>(work_area[i][j].y * height / 100.0f));
    }

    return wa;
  }

  Workflow::Workflow(const userver::components::ComponentConfig& config,
    const userver::components::ComponentContext& context)
    : LoggableComponentBase{config, context},
      task_processor_(context.GetTaskProcessor(config["task_processor"].As<std::string>())),
      fs_task_processor_(context.GetTaskProcessor(config["fs-task-processor"].As<std::string>())),
      http_client_(context.FindComponent<userver::components::HttpClient>().GetHttpClient()),
      vstreams_config_cache_(context.FindComponent<VStreamsConfigCache>()),
      pg_cluster_(context.FindComponent<userver::components::Postgres>(kDatabase).GetCluster()),
      logger_(context.FindComponent<userver::components::Logging>().GetLogger(std::string(kLogger)))
  {
    local_config_.allow_group_id_without_auth = config[ConfigParams::SECTION_NAME][ConfigParams::ALLOW_GROUP_ID_WITHOUT_AUTH].As<decltype(local_config_.allow_group_id_without_auth)>();
    local_config_.ban_maintenance_interval = config[ConfigParams::SECTION_NAME][ConfigParams::BAN_MAINTENANCE_INTERVAL].As<decltype(local_config_.ban_maintenance_interval)>();
    local_config_.events_log_maintenance_interval = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_LOG_MAINTENANCE_INTERVAL].As<decltype(local_config_.events_log_maintenance_interval)>();
    local_config_.events_log_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_LOG_TTL].As<decltype(local_config_.events_log_ttl)>();
    local_config_.events_screenshots_path = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_SCREENSHOTS_PATH].As<decltype(local_config_.events_screenshots_path)>();
    if (!local_config_.events_screenshots_path.empty() && !local_config_.events_screenshots_path.ends_with("/"))
      local_config_.events_screenshots_path += "/";
    local_config_.events_screenshots_url_prefix = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_SCREENSHOTS_URL_PREFIX].As<decltype(local_config_.events_screenshots_url_prefix)>();
    if (!local_config_.events_screenshots_url_prefix.empty() && !local_config_.events_screenshots_url_prefix.ends_with("/"))
      local_config_.events_screenshots_url_prefix += "/";
    local_config_.failed_path = config[ConfigParams::SECTION_NAME][ConfigParams::FAILED_PATH].As<decltype(local_config_.failed_path)>();
    if (!local_config_.failed_path.empty() && !local_config_.failed_path.ends_with("/"))
      local_config_.failed_path += "/";
    local_config_.failed_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::FAILED_TTL].As<decltype(local_config_.failed_ttl)>();

    if (local_config_.ban_maintenance_interval.count() > 0)
      ban_maintenance_task_.Start(kBanMaintenanceName,
        {std::chrono::milliseconds(local_config_.ban_maintenance_interval),
          {userver::utils::PeriodicTask::Flags::kStrong}},
        [this]
        { doBanMaintenance(); });

    if (local_config_.events_log_maintenance_interval.count() > 0)
      events_log_maintenance_task_.Start(kEventsLogMaintenanceName,
        {std::chrono::milliseconds(local_config_.events_log_maintenance_interval),
          {userver::utils::PeriodicTask::Flags::kStrong}},
        [this]
        { doEventsLogMaintenance(); });
  }

  userver::yaml_config::Schema Workflow::GetStaticConfigSchema()
  {
    return userver::yaml_config::MergeSchemas<LoggableComponentBase>(R"~(
# yaml
type: object
description: Component for license plate recognition workflow
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
            ban-maintenance-interval:
                type: string
                description: Interval in for ban maintenance
                defaultDescription: 5s
            events-log-maintenance-interval:
                type: string
                description: Interval for events log maintenance
                defaultDescription: 2h
            events-log-ttl:
                type: string
                description: Time to live for events log
                defaultDescription: 4h
            screenshots-path:
                type: string
                description: Local path for saving event screenshots
                defaultDescription: '/opt/falprs/static/frs/screenshots/'
            screenshots-url-prefix:
                type: string
                description: Web URL prefix for events' screenshots
                defaultDescription: 'http://localhost:9051/lprs/'
            failed-path:
                type: string
                description: Local path for saving unrecognized license plates screenshots
                defaultDescription: '/opt/falprs/static/lprs/failed/'
            failed-ttl:
                type: string
                description: Time to live for the unrecognized license plates screenshots
                defaultDescription: 60d
  )~");
  }

  void Workflow::startWorkflow(std::string&& vstream_key)
  {
    std::chrono::milliseconds workflow_timeout{std::chrono::seconds{0}};
    // scope for accessing cache
    {
      const auto cache = vstreams_config_cache_.Get();
      if (!cache->getData().contains(vstream_key))
        return;

      workflow_timeout = cache->getData().at(vstream_key).workflow_timeout;
    }

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
      tasks_.Detach(AsyncNoSpan(task_processor_, &Workflow::processPipeline, this, std::move(vstream_key)));
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

  const LocalConfig& Workflow::getLocalConfig()
  {
    return local_config_;
  }

  const userver::logging::LoggerPtr& Workflow::getLogger()
  {
    return logger_;
  }

  // private methods
  void Workflow::OnAllComponentsAreStopping()
  {
    tasks_.CancelAndWait();
  }

  void Workflow::processPipeline(std::string&& vstream_key)
  {
    VStreamConfig config;

    // scope for accessing cache
    {
      auto cache = vstreams_config_cache_.Get();
      if (!cache->getData().contains(vstream_key))
      {
        stopWorkflow(std::move(vstream_key));
        return;
      }

      config = cache->getData().at(vstream_key);
    }

    if (config.screenshot_url.empty())
    {
      stopWorkflow(std::move(vstream_key));
      return;
    }

    if (config.logs_level <= userver::logging::Level::kDebug)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug)
        << "Start processPipeline: vstream_key = " << vstream_key
        << absl::Substitute(";  frame_url = $0", config.screenshot_url);

    try
    {
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  before image acquisition";

      // parse user and password
      std::string auth_user;
      std::string auth_password;
      if (auto char_alpha = config.screenshot_url.find('@'); char_alpha != std::string::npos)
      {
        if (auto protocol_suffix = config.screenshot_url.find("://"); protocol_suffix != std::string::npos && protocol_suffix < char_alpha)
        {
          if (auto char_colon = config.screenshot_url.find(':', protocol_suffix + 3); char_colon != std::string::npos && char_colon < char_alpha)
          {
            auto char_slash = protocol_suffix + 2;
            auth_user = config.screenshot_url.substr(char_slash + 1, char_colon - char_slash - 1);
            auth_password = config.screenshot_url.substr(char_colon + 1, char_alpha - char_colon - 1);
          }
        }
      }
      // clang-format off
      auto capture_response = http_client_.CreateRequest()
        .get(config.screenshot_url)
        .http_auth_type(userver::clients::http::HttpAuthType::kAnySafe, false, auth_user, auth_password)
        .retry(config.max_capture_error_count)
        .timeout(config.capture_timeout)
        .perform();
      // clang-format on
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  after image acquisition";

      if (capture_response->status_code() != userver::clients::http::Status::OK || capture_response->body_view().empty())
      {
        if (config.logs_level <= userver::logging::Level::kError)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError)
            << "vstream_key = " << vstream_key
            << ";  url = " << config.screenshot_url
            << ";  status_code = " << capture_response->status_code();
        if (config.delay_after_error.count() > 0)
        {
          if (config.logs_level <= userver::logging::Level::kError)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError)
              << "vstream_key = " << vstream_key
              << ";  delay for " << config.delay_after_error.count() << "ms";
          nextPipeline(std::move(vstream_key), config.delay_after_error);
        } else
          stopWorkflow(std::move(vstream_key));

        return;
      }

      if (config.logs_level <= userver::logging::Level::kTrace)
      {
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  image size " << capture_response->body_view().size();
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  before decoding the image";
      }
      cv::Mat frame = imdecode(std::vector<char>(capture_response->body_view().begin(), capture_response->body_view().end()),
        cv::IMREAD_COLOR);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  after decoding the image";

      // for test: rotate image
      /*float angle = -12.0f;
      cv::Point2f p_center((frame.cols - 1) / 2.0f, (frame.rows - 1) / 2.0f);
      cv::Mat m_rotation = cv::getRotationMatrix2D(p_center, angle, 1.0);
      cv::warpAffine(frame, frame, m_rotation, frame.size());
      cv::imwrite("test.jpg", frame); */

      // cv::Mat frame = cv::imread("2023-05-16_17_43_57.png", cv::IMREAD_COLOR);
      // cv::Mat frame = cv::imread("ru002.jpg", cv::IMREAD_COLOR);
      std::vector<Vehicle> detected_vehicles;
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  before doInferenceVdNet";
      doInferenceVdNet(frame, config, detected_vehicles);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  after doInferenceVdNet";

      if (config.flag_process_special)
      {
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
            << "vstream_key = " << vstream_key
            << ";  before doInferenceVcNet";
        doInferenceVcNet(frame, config, detected_vehicles);
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
            << "vstream_key = " << vstream_key
            << ";  after doInferenceVcNet";
      }

      if (config.logs_level <= userver::logging::Level::kTrace)
        for (size_t i = 0; i < detected_vehicles.size(); ++i)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  vehicle " << i  << " confidence: " << detected_vehicles[i].confidence;

      // check for special vehicles ban
      bool is_special_banned = false;
      {
        auto ban_special_data_ptr = ban_special_data.Lock();
        if (ban_special_data_ptr->contains(vstream_key))
        {
          auto now = std::chrono::steady_clock::now();
          is_special_banned = (*ban_special_data_ptr)[vstream_key] > now;
          if (is_special_banned)
          {
            if (config.logs_level <= userver::logging::Level::kTrace)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                << "vstream_key = " << vstream_key
                << ";  special vehicles are banned ("
                << std::chrono::duration_cast<std::chrono::seconds>((*ban_special_data_ptr)[vstream_key] - now) << " left)";
          } else
          {
            ban_special_data_ptr->erase(vstream_key);
            if (config.logs_level <= userver::logging::Level::kTrace)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                << "vstream_key = " << vstream_key
                << ";  special vehicles are no longer banned";
          }
        }
      }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  before doInferenceLpdNet";
      doInferenceLpdNet(frame, config, detected_vehicles);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << vstream_key
          << ";  after doInferenceLpdNet";

      removeDuplicatePlates(config, detected_vehicles, frame.cols, frame.rows);

      // for test
      // save images of the special vehicles
      /*for (size_t v = 0;  v < detected_vehicles.size(); ++v)
        if (detected_vehicles[v].is_special)
        {
          cv::Rect roi(cv::Point{static_cast<int>(detected_vehicles[v].bbox[0]), static_cast<int>(detected_vehicles[v].bbox[1])},
            cv::Point{static_cast<int>(detected_vehicles[v].bbox[2]), static_cast<int>(detected_vehicles[v].bbox[3])});
          imwrite(absl::Substitute("special_$0_$1.jpg",  config.id_vstream, v), frame(roi));
        }*/

      std::vector<LicensePlate*> detected_plates;
      for (const auto& [bbox, confidence, is_special, license_plates] : detected_vehicles)
        for (const auto& license_plate : license_plates)
          detected_plates.push_back(const_cast<LicensePlate*>(&license_plate));

      bool result = true;
      if (!detected_plates.empty())
      {
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
            << "vstream_key = " << vstream_key
            << ";  before doInferenceLprNet";
        result = doInferenceLprNet(frame, config, detected_plates);
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
            << "vstream_key = " << vstream_key
            << ";  after doInferenceLprNet";
      }
      if (result)
      {
        auto now = std::chrono::steady_clock::now();
        userver::formats::json::ValueBuilder json_data;
        userver::formats::json::ValueBuilder callback_info;
        bool has_special = false;
        bool has_failed = false;
        for (const auto& [bbox, confidence, is_special, license_plates] : detected_vehicles)
        {
          userver::formats::json::ValueBuilder vehicle_data;
          has_special = has_special || is_special;
          for (const auto& [bbox, confidence, kpts, plate_class, plate_numbers] : license_plates)
          {
            has_failed = has_failed || plate_numbers.empty();
            for (const auto& [number, score] : plate_numbers)
            {
              auto k = absl::StrCat(vstream_key, "_", number);

              // Description of the two-stage ban.
              // If the system sees the number for the first time, then after processing it will fall into the first stage of the ban.
              // At the first stage of the ban, the number is ignored regardless of its location in the frame.
              // After some time (config ban-duration), the number falls into the second stage of the ban.
              // At the second stage of the ban (config ban-duration-area), the number is also ignored until it changes its location in the frame.
              // If at the second stage of the ban the number changes its location (config ban-iou-threshold), it will be processed and will again be included in the first stage.
              if (config.ban_duration.count() > 0 && config.ban_duration_area.count() > 0)
              {
                bool is_banned = false;
                auto banned_tp1 = now + config.ban_duration;
                auto banned_tp2 = now + config.ban_duration_area;
                auto banned_bbox = cv::Rect2f(cv::Point2f{bbox[0], bbox[1]}, cv::Point2f{bbox[2], bbox[3]});
                auto data_ptr = ban_data.Lock();
                if (data_ptr->contains(k))
                {
                  is_banned = (*data_ptr)[k].tp1 > now;
                  if (is_banned)
                  {
                    if (config.logs_level <= userver::logging::Level::kDebug)
                      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug)
                        << "vstream_key = " << vstream_key
                        << ";  plate number " << number << " is banned at the first stage ("
                        << std::chrono::duration_cast<std::chrono::seconds>((*data_ptr)[k].tp1 - now) << " left)";
                  } else
                  {
                    // check area ban
                    auto iou_value = iou(banned_bbox, (*data_ptr)[k].bbox);
                    is_banned = iou_value > config.ban_iou_threshold;
                    if (is_banned)
                    {
                      // extending the second stage of the ban
                      banned_bbox = (*data_ptr)[k].bbox;
                      banned_tp1 = (*data_ptr)[k].tp1;
                    }
                    if (config.logs_level <= userver::logging::Level::kDebug)
                    {
                      if (is_banned)
                        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug)
                          << absl::StrFormat("vstream_key = %s;  plate_number %s is banned at the second stage (iou = %.2f, threshold = %.2f)",
                               vstream_key, number, iou_value, config.ban_iou_threshold);
                      else
                        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug)
                          << absl::StrFormat("vstream_key = %s;  plate_numbers %s was removed from the the second stage ban (iou = %.2f, threshold = %.2f)",
                               vstream_key, number, iou_value, config.ban_iou_threshold);
                    }
                  }
                }
                // update ban data
                (*data_ptr)[k] = {banned_tp1, banned_tp2, banned_bbox};
                if (is_banned)
                  continue;
              }

              if (config.logs_level <= userver::logging::Level::kInfo)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo)
                  << "vstream_key = " << vstream_key
                  << ";  plate number: " << number;
              userver::formats::json::ValueBuilder plate_data;
              plate_data[Api::PARAM_BOX] = userver::formats::json::MakeArray(
                static_cast<int32_t>(bbox[0]),
                static_cast<int32_t>(bbox[1]),
                static_cast<int32_t>(bbox[2]),
                static_cast<int32_t>(bbox[3]));
              plate_data[Api::PARAM_KPTS] = userver::formats::json::MakeArray(
                static_cast<int32_t>(kpts[0]),
                static_cast<int32_t>(kpts[1]),
                static_cast<int32_t>(kpts[2]),
                static_cast<int32_t>(kpts[3]),
                static_cast<int32_t>(kpts[4]),
                static_cast<int32_t>(kpts[5]),
                static_cast<int32_t>(kpts[6]),
                static_cast<int32_t>(kpts[7]));
              plate_data[Api::PARAM_NUMBER] = number;
              plate_data[Api::PARAM_SCORE] = score;
              plate_data[Api::PARAM_PLATE_TYPE] = PLATE_CLASSES[plate_class];
              vehicle_data[Api::PARAM_PLATES_INFO].PushBack(std::move(plate_data));
              userver::formats::json::ValueBuilder plate_data_short;
              plate_data_short[Api::PARAM_PLATE_TYPE] = PLATE_CLASSES[plate_class];
              plate_data_short[Api::PARAM_NUMBER] = number;
              callback_info.PushBack(std::move(plate_data_short));
            }
          }

          if (!vehicle_data.IsEmpty() || (is_special && !is_special_banned))
          {
            vehicle_data[Api::PARAM_IS_SPECIAL] = is_special;
            vehicle_data[Api::PARAM_CONFIDENCE] = confidence;
            vehicle_data[Api::PARAM_BOX] = userver::formats::json::MakeArray(static_cast<int32_t>(bbox[0]),
              static_cast<int32_t>(bbox[1]),
              static_cast<int32_t>(bbox[2]),
              static_cast<int32_t>(bbox[3]));
            json_data[Api::PARAM_VEHICLES_INFO].PushBack(std::move(vehicle_data));
          }
        }

        if (!json_data.IsEmpty())
        {
          auto t_now = std::chrono::system_clock::now();
          auto log_date = userver::storages::postgres::TimePointTz{t_now};
          auto uuid = boost::uuids::to_string(boost::uuids::random_generator()());
          auto path_suffix = absl::Substitute("$0/$1/$2/$3/", uuid[0], uuid[1], uuid[2], uuid[3]);
          auto screenshot_extension = ".jpg";
          json_data[Api::PARAM_SCREENSHOT_URL] = absl::StrCat(local_config_.events_screenshots_url_prefix, path_suffix,
            uuid, screenshot_extension);
          json_data[Api::PARAM_EVENT_DATE] = log_date;
          auto id_event = addEventLog(config.id_vstream, log_date, json_data.ExtractValue());

          // write screenshot to a file
          auto path_prefix = absl::StrCat(local_config_.events_screenshots_path, path_suffix);
          userver::fs::CreateDirectories(fs_task_processor_, path_prefix);
          auto path = absl::StrCat(path_prefix, uuid, screenshot_extension);
          userver::fs::RewriteFileContents(fs_task_processor_, path, capture_response->body_view());
          userver::fs::Chmod(fs_task_processor_, path,
            boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write | boost::filesystem::perms::others_read | boost::filesystem::perms::others_write);

          // send data to callback
          userver::formats::json::ValueBuilder json_callback;
          json_callback[Api::PARAM_STREAM_ID] = config.ext_id;
          json_callback[Api::PARAM_EVENT_DATE] = log_date;
          json_callback[Api::PARAM_EVENT_ID] = id_event;
          if (!callback_info.IsNull())
            json_callback[Api::PARAM_PLATES_INFO] = callback_info;
          json_callback[Api::PARAM_HAS_SPECIAL] = has_special;
          try
          {
            // clang-format off
            auto response = http_client_.CreateRequest()
              .post(config.callback_url)
              .headers({{userver::http::headers::kContentType, userver::http::content_type::kApplicationJson.ToString()}})
              .data(userver::formats::json::ToString(json_callback.ExtractValue()))
              .timeout(config.callback_timeout)
              .perform();
            // clang-format on
            if (!(response->status_code() == userver::clients::http::Status::OK
                  || response->status_code() == userver::clients::http::Status::NoContent))
              if (config.logs_level <= userver::logging::Level::kWarning)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kWarning)
                  << "vstream_key = " << vstream_key
                  << "; error sending data to callback " << config.callback_url;
          } catch (std::exception& e)
          {
            LOG_ERROR_TO(logger_)
              << "vstream_key = " << vstream_key
              << "; error sending data to callback " << config.callback_url
              << "; " << e.what();
          }
        }

        if (has_special)
        {
          auto ban_special_data_ptr = ban_special_data.Lock();
          (*ban_special_data_ptr)[vstream_key] = now + config.ban_duration;
        }

        if (has_failed && config.flag_save_failed)
        {
          auto uuid = boost::uuids::to_string(boost::uuids::random_generator()());
          auto path_suffix = absl::Substitute("$0/", config.ext_id);
          auto screenshot_extension = ".jpg";

          // write failed screenshot to a file
          auto path_prefix = absl::StrCat(local_config_.failed_path, path_suffix);
          userver::fs::CreateDirectories(fs_task_processor_, path_prefix);
          auto path = absl::StrCat(path_prefix, uuid, screenshot_extension);
          auto path_draw = absl::StrCat(path_prefix, uuid, "_draw", screenshot_extension);
          userver::fs::RewriteFileContents(fs_task_processor_, path, capture_response->body_view());
          userver::fs::Chmod(fs_task_processor_, path,
            boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write | boost::filesystem::perms::others_read | boost::filesystem::perms::others_write);

          // draw failed license plates on the frame
          if (!config.work_area.empty())
          {
            auto wa = convertToAbsolute(config.work_area, frame.cols, frame.rows);
            polylines(frame, wa, true, cv::Scalar(0, 200, 0), 2);
          }
          for (const auto& [bbox, confidence, is_special, license_plates] : detected_vehicles)
          {
            std::vector<std::vector<cv::Point>> vehicle_polygons;
            vehicle_polygons.push_back({{static_cast<int>(bbox[0]), static_cast<int>(bbox[1])},
              {static_cast<int>(bbox[2]), static_cast<int>(bbox[1])},
              {static_cast<int>(bbox[2]), static_cast<int>(bbox[3])},
              {static_cast<int>(bbox[0]), static_cast<int>(bbox[3])}});
            polylines(frame, vehicle_polygons, true, is_special ? cv::Scalar(0, 0, 200) : cv::Scalar(200, 0, 0), 2);
          }
          std::vector<std::vector<cv::Point>> plate_polygons_good;
          plate_polygons_good.reserve(detected_plates.size());
          std::vector<std::vector<cv::Point>> plate_polygons_failed;
          plate_polygons_failed.reserve(detected_plates.size());
          for (auto plate_ptr : detected_plates)
          {
            auto& [bbox, confidence, kpts, plate_class, plate_numbers] = *plate_ptr;
            std::vector<cv::Point> p = {{static_cast<int>(kpts[0]), static_cast<int>(kpts[1])},
                {static_cast<int>(kpts[2]), static_cast<int>(kpts[3])},
                {static_cast<int>(kpts[4]), static_cast<int>(kpts[5])},
                {static_cast<int>(kpts[6]), static_cast<int>(kpts[7])}};
            if (plate_ptr->plate_numbers.empty())
              plate_polygons_failed.push_back(p);
            else
              plate_polygons_good.push_back(p);
          }
          if (!plate_polygons_good.empty())
            polylines(frame, plate_polygons_good, true, cv::Scalar(2, 105, 255), 2);
          if (!plate_polygons_failed.empty())
            polylines(frame, plate_polygons_failed, true, cv::Scalar(226, 43, 138), 2);
          imwrite(path_draw, frame);
        }
      }

      // for test draw on the frame
      /*if (!config.work_area.empty())
      {
        auto wa = convertToAbsolute(config.work_area, frame.cols, frame.rows);
        polylines(frame, wa, true, cv::Scalar(0, 200, 0), 2);
      }
      for (const auto& vehicle : detected_vehicles)
      {
        std::vector<std::vector<cv::Point>> vehicle_polygons;
        vehicle_polygons.push_back({{static_cast<int>(vehicle.bbox[0]), static_cast<int>(vehicle.bbox[1])},
          {static_cast<int>(vehicle.bbox[2]), static_cast<int>(vehicle.bbox[1])},
          {static_cast<int>(vehicle.bbox[2]), static_cast<int>(vehicle.bbox[3])},
          {static_cast<int>(vehicle.bbox[0]), static_cast<int>(vehicle.bbox[3])}});
        polylines(frame, vehicle_polygons, true, vehicle.is_special ? cv::Scalar(0, 0, 200) : cv::Scalar(200, 0, 0), 2);
      }
      std::vector<std::vector<cv::Point>> plate_polygons;
      plate_polygons.reserve(detected_plates.size());
      for (auto plate_ptr : detected_plates)
      {
        auto& plate = *plate_ptr;
        plate_polygons.push_back({{static_cast<int>(plate.kpts[0]), static_cast<int>(plate.kpts[1])},
          {static_cast<int>(plate.kpts[2]), static_cast<int>(plate.kpts[3])},
          {static_cast<int>(plate.kpts[4]), static_cast<int>(plate.kpts[5])},
          {static_cast<int>(plate.kpts[6]), static_cast<int>(plate.kpts[7])}});
      }
      if (!plate_polygons.empty())
        polylines(frame, plate_polygons, true, cv::Scalar(2, 102, 255), 2);
      imwrite("test.jpg", frame);*/
    } catch (std::exception& e)
    {
      if (config.logs_level <= userver::logging::Level::kError)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError)
          << "vstream_key = " << vstream_key
          << ";  " << e.what();

      if (config.delay_after_error.count() > 0)
      {
        if (config.logs_level <= userver::logging::Level::kError)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError)
            << "vstream_key = " << vstream_key
            << ";  delay for " << config.delay_after_error.count() << "ms";
        nextPipeline(std::move(vstream_key), config.delay_after_error);
      } else
        stopWorkflow(std::move(vstream_key));

      return;
    }

    if (config.logs_level <= userver::logging::Level::kDebug)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug)
        << "End processPipeline: vstream_key = " << vstream_key << ";";

    nextPipeline(std::move(vstream_key), config.delay_between_frames);
  }

  void Workflow::doBanMaintenance()
  {
    LOG_DEBUG_TO(logger_) << "call doBanMaintenance";
    auto t_now = std::chrono::steady_clock::now();
    auto data_ptr = ban_data.Lock();
    absl::erase_if(*data_ptr,
      [&t_now](const auto& item)
      {
        return item.second.tp2 < t_now;
      });
  }

  void Workflow::doEventsLogMaintenance() const
  {
    LOG_DEBUG_TO(logger_) << "call doEventsLogMaintenance";
    auto tp = std::chrono::system_clock::now() - local_config_.events_log_ttl;
    LOG_DEBUG_TO(logger_) << "delete event logs older than " << tp;
    const userver::storages::postgres::Query query{SQL_REMOVE_OLD_EVENTS};
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      trx.Execute(query, userver::storages::postgres::TimePointTz{tp});
      trx.Commit();
    } catch (std::exception& e)
    {
      trx.Rollback();
      LOG_ERROR_TO(logger_) << e.what();

      return;
    }

    // delete old screenshot files
    const HashSet<std::string> img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tiff"};
    if (std::filesystem::exists(local_config_.events_screenshots_path))
      for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(local_config_.events_screenshots_path))
        if (dir_entry.is_regular_file() && img_extensions.contains(dir_entry.path().extension().string()))
        {
          if (auto t_file = std::chrono::file_clock::to_sys(dir_entry.last_write_time()); t_file < tp)
            std::filesystem::remove(dir_entry);
        }

    // delete old failed license plates screenshot files
    tp = std::chrono::system_clock::now() - local_config_.failed_ttl;
    if (std::filesystem::exists(local_config_.failed_path))
      for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(local_config_.failed_path))
        if (dir_entry.is_regular_file() && img_extensions.contains(dir_entry.path().extension().string()))
        {
          if (auto t_file = std::chrono::file_clock::to_sys(dir_entry.last_write_time()); t_file < tp)
            std::filesystem::remove(dir_entry);
        }
  }

  void Workflow::nextPipeline(std::string&& vstream_key, const std::chrono::milliseconds delay)
  {
    userver::engine::InterruptibleSleepFor(delay);

    bool do_next = false;
    bool is_timeout = false;

    // scope for accessing concurrent variable
    {
      const auto now = std::chrono::steady_clock::now();
      auto data_ptr = vstream_timeouts.Lock();
      if (data_ptr->contains(vstream_key))
        if (data_ptr->at(vstream_key) < now)
        {
          data_ptr->erase(vstream_key);
          is_timeout = true;
        }
    }

    // scope for accessing concurrent variable
    {
      auto data_ptr = being_processed_vstreams.Lock();
      if (data_ptr->contains(vstream_key))
      {
        if (data_ptr->at(vstream_key) && !is_timeout)
          do_next = true;
        else
          data_ptr->erase(vstream_key);
      }
    }

    if (is_timeout)
      LOG_INFO_TO(logger_) << "Stopping a workflow by timeout: vstream_key = " << vstream_key << ";";

    if (do_next)
      tasks_.Detach(AsyncNoSpan(task_processor_, &Workflow::processPipeline, this, std::move(vstream_key)));
  }

  // Inference pipeline methods
  std::vector<float> Workflow::preprocessImageForVdNet(const cv::Mat& img, const int32_t width, const int32_t height, cv::Point2f& shift,
    double& scale)
  {
    return preprocessImageForLpdNet(img, width, height, shift, scale);
  }

    std::vector<float> Workflow::preprocessImageForVcNet(const cv::Mat& img, const int32_t width, const int32_t height)
  {
    cv::Mat out(height, width, CV_8UC3);
    resize(img, out, out.size(), 0, 0, cv::INTER_AREA);

    constexpr int32_t channels = 3;
    const int32_t input_size = channels * width * height;
    std::vector<float> input_buffer(input_size);
    /*const std::vector<float> means = {0.485, 0.456, 0.406};
    const std::vector<float> std_d = {0.229, 0.224, 0.225};*/
    const std::vector<float> means = {0.5, 0.5, 0.5};
    const std::vector<float> std_d = {0.5, 0.5, 0.5};
    for (auto c = 0; c < channels; ++c)
      for (auto h = 0; h < height; ++h)
        for (auto w = 0; w < width; ++w)
          input_buffer[c * height * width + h * width + w] =
            (static_cast<float>(out.at<cv::Vec3b>(h, w)[2 - c]) / 255.0f - means[c]) / std_d[c];

    return input_buffer;
  }

  std::vector<float> Workflow::preprocessImageForLpdNet(const cv::Mat& img, const int32_t width, const int32_t height, cv::Point2f& shift,
    double& scale)
  {
    const auto r_w = width / (img.cols * 1.0);
    const auto r_h = height / (img.rows * 1.0);
    scale = fmin(r_w, r_h);
    const auto ww = static_cast<int>(lround(scale * img.cols));
    const auto hh = static_cast<int>(lround(scale * img.rows));
    shift.x = static_cast<float>(width - ww) / 2;
    shift.y = static_cast<float>(height - hh) / 2;
    cv::Mat re(hh, ww, CV_8UC3);
    resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

    const int border_top = static_cast<int>(shift.y);
    const int border_bottom = height - hh - border_top;
    const int border_left = static_cast<int>(shift.x);
    const int border_right = width - ww - border_left;

    cv::Mat out;
    copyMakeBorder(re, out, border_top, border_bottom,
      border_left, border_right, cv::BORDER_CONSTANT, {114, 114, 114});

    /*cv::Mat out(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(shift.x, shift.y, re.cols, re.rows)));*/

    // for test
    // cv::imwrite(absl::Substitute("p_$0_$1.jpg", border_left, border_top), out);

    constexpr int32_t channels = 3;
    const int32_t input_size = channels * width * height;
    std::vector<float> input_buffer(input_size);
    for (auto c = 0; c < channels; ++c)
      for (auto h = 0; h < height; ++h)
        for (auto w = 0; w < width; ++w)
          input_buffer[c * height * width + h * width + w] =
            static_cast<float>(out.at<cv::Vec3b>(h, w)[2 - c]) / 255.0f;

    return input_buffer;
  }

  std::vector<float> Workflow::preprocessImageForLprNet(const cv::Mat& img, const int32_t width, const int32_t height, cv::Point2f& shift, double& scale)
  {
    const auto r_w = width / (img.cols * 1.0);
    const auto r_h = height / (img.rows * 1.0);
    scale = fmin(r_w, r_h);
    const auto ww = static_cast<int>(lround(scale * img.cols));
    const auto hh = static_cast<int>(lround(scale * img.rows));
    shift.x = static_cast<float>(width - ww) / 2;
    shift.y = static_cast<float>(height - hh) / 2;
    cv::Mat re(hh, ww, CV_8UC3);
    resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat out;
    copyMakeBorder(re, out, static_cast<int>(shift.y), static_cast<int>(shift.y),
      static_cast<int>(shift.x), static_cast<int>(shift.x), cv::BORDER_CONSTANT, {114, 114, 114});

    /*cv::Mat out(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(shift.x, shift.y, re.cols, re.rows)));*/

    // for test
    // cv::imwrite("plate.jpg", out);

    constexpr int32_t channels = 3;
    const int32_t input_size = channels * width * height;
    std::vector<float> input_buffer(input_size);
    for (auto c = 0; c < channels; ++c)
      for (auto h = 0; h < height; ++h)
        for (auto w = 0; w < width; ++w)
          input_buffer[c * height * width + h * width + w] =
            static_cast<float>(out.at<cv::Vec3b>(h, w)[2 - c]) / 255.0f;

    return input_buffer;
  }

  bool Workflow::doInferenceVdNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles) const
  {
    detected_vehicles.clear();

    std::unique_ptr<tc::InferenceServerHttpClient> triton_client;
    auto err = tc::InferenceServerHttpClient::Create(&triton_client, config.vd_net_inference_server, false);
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create inference client: $0", err.Message());
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  before preprocess image for VDNet";
    cv::Point2f shift;
    double scale;
    auto input_buffer = preprocessImageForVdNet(img, config.lpd_net_input_width, config.lpd_net_input_height, shift,
      scale);
    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  after preprocess image for VDNet";

    std::vector<uint8_t> input_data(config.vd_net_input_width * config.vd_net_input_height * 3 * sizeof(float));
    memcpy(input_data.data(), input_buffer.data(), input_data.size());
    std::vector<int64_t> shape = {1, 3, config.vd_net_input_height, config.vd_net_input_width};
    tc::InferInput* input;
    err = tc::InferInput::Create(&input, config.vd_net_input_tensor_name, shape, "FP32");
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create input data: $0", err.Message());
      return false;
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);

    tc::InferRequestedOutput* output;
    err = tc::InferRequestedOutput::Create(&output, config.vd_net_output_tensor_name);
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create output data: $0", err.Message());
      return false;
    }
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

    std::vector inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};
    err = input_ptr->AppendRaw(input_data);
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to set up input data: $0", err.Message());
      return false;
    }

    tc::InferOptions options(config.vd_net_model_name);
    options.model_version_ = "";
    tc::InferResult* result;

    AsyncNoSpan(fs_task_processor_,
      [&]
      {
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
            << "vstream_key = " << config.id_group << "_" << config.ext_id
            << ";  before inference VDNet";
        err = triton_client->Infer(&result, options, inputs, outputs);
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
            << "vstream_key = " << config.id_group << "_" << config.ext_id
            << ";  after inference VDNet";
      })
      .Get();
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to send inference request: $0", err.Message());
      return false;
    }

    std::shared_ptr<tc::InferResult> result_ptr(result);
    if (!result_ptr->RequestStatus().IsOk())
    {
      LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to receive inference result: $0", err.Message());
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kDebug)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  inference VDNet OK";

    const float* data;
    size_t data_size;
    result_ptr->RawData(config.vd_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

    // the output tensor has a dimension of [7, 8400]
    //  0 - bbox x_center
    //  1 - bbox y_center
    //  2 - bbox width
    //  3 - bbox height
    //  4 - confidence of class 0
    //  5 - confidence of class 1
    //  6 - confidence of class 2
    auto num_cols = 8400;
    auto bbox_index = 0;
    auto class_start_index = bbox_index + 4;
    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  vehicle confidence threshold: " << config.vehicle_confidence;
    for (auto j = 0; j < num_cols; ++j)
    {
      auto k = class_start_index;
      if (auto confidence = data[k * num_cols + j]; confidence > config.vehicle_confidence)
      {
        auto xmin = std::fmax(static_cast<float>((data[(bbox_index + 0) * num_cols + j] - data[(bbox_index + 2) * num_cols + j] / 2 - shift.x) / scale), 0.0f);
        auto ymin = std::fmax(static_cast<float>((data[(bbox_index + 1) * num_cols + j] - data[(bbox_index + 3) * num_cols + j] / 2 - shift.y) / scale), 0.0f);
        auto xmax = std::fmin(static_cast<float>((data[(bbox_index + 0) * num_cols + j] + data[(bbox_index + 2) * num_cols + j] / 2 - shift.x) / scale), static_cast<float>(img.cols - 1));
        auto ymax = std::fmin(static_cast<float>((data[(bbox_index + 1) * num_cols + j] + data[(bbox_index + 3) * num_cols + j] / 2 - shift.y) / scale), static_cast<float>(img.rows - 1));

        // remove small vehicle detections
        auto vehicle_area = (xmax - xmin + 1) * (ymax - ymin + 1);
        if (auto screen_area = static_cast<float>(img.cols * img.rows); vehicle_area / screen_area < config.vehicle_area_ratio_threshold)
          continue;

        detected_vehicles.emplace_back();
        detected_vehicles.back().bbox[0] = xmin;
        detected_vehicles.back().bbox[1] = ymin;
        detected_vehicles.back().bbox[2] = xmax;
        detected_vehicles.back().bbox[3] = ymax;
        detected_vehicles.back().confidence = confidence;
      }
    }

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  before nms_vehicles count: " << detected_vehicles.size();
    nms_vehicles(detected_vehicles, config.vehicle_iou_threshold);
    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  after nms_vehicles count: " << detected_vehicles.size();

    return true;
  }

  bool Workflow::doInferenceVcNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles) const
  {
    std::vector<userver::engine::TaskWithResult<triton::client::Error>> tasks;
    tasks.reserve(detected_vehicles.size());

    std::vector<std::unique_ptr<tc::InferenceServerHttpClient>> triton_clients;
    triton_clients.resize(detected_vehicles.size());

    std::vector<tc::InferResult*> results;
    results.resize(detected_vehicles.size());

    std::vector<std::vector<uint8_t>> inputs_data;
    inputs_data.resize(detected_vehicles.size());

    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
    input_ptrs.reserve(detected_vehicles.size());

    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_ptrs;
    output_ptrs.reserve(detected_vehicles.size());

    std::vector<tc::InferOptions> options;
    options.reserve(detected_vehicles.size());

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  before inference VcNet";

    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[vindex], config.lpr_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create inference client: $0", err.Message());
        return false;
      }

      const auto& [bbox, confidence, is_special, license_plates] = detected_vehicles[vindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  before preprocess image " << vindex << " for VcNet";
      cv::Rect roi(cv::Point{static_cast<int>(bbox[0]), static_cast<int>(bbox[1])},
        cv::Point{static_cast<int>(bbox[2]), static_cast<int>(bbox[3])});
      auto input_buffer = preprocessImageForVcNet(img(roi), config.vc_net_input_width, config.vc_net_input_height);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  after preprocess image " << vindex << " for VcNet";

      inputs_data[vindex].resize(config.vc_net_input_width * config.vc_net_input_height * 3 * sizeof(float));
      memcpy(inputs_data[vindex].data(), input_buffer.data(), inputs_data[vindex].size());
      std::vector<int64_t> shape = {1, 3, config.vc_net_input_height, config.vc_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.vc_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create input data: $0", err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.vc_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create output data: $0", err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[vindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to set up input data: $0", err.Message());
        return false;
      }
      options.emplace_back(config.vc_net_model_name);
      options.back().model_version_ = "";

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&, vindex]
        {
          return triton_clients[vindex]->Infer(&results[vindex], options[vindex], {input_ptrs[vindex].get()}, {output_ptrs[vindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  after inference VcNet";

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  special confidence threshold: " << config.special_confidence;

    bool is_ok = false;
    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      if (auto err = tasks[vindex].Get(); !err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to send inference request (vindex = $0): $1", vindex, err.Message());
        continue;
      }

      const std::shared_ptr<tc::InferResult> result_ptr(results[vindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to receive inference result (vindex = $0): $1", vindex, result_ptr->RequestStatus().Message());
        continue;
      }

      const float* data;
      size_t data_size;
      result_ptr->RawData(config.vc_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      std::vector<float> scores;
      scores.assign(data, data + 2);

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  vindex = " << vindex
          << ";  softmax scores:" << vindex
          << " scores[0]: " << scores[0]
          << " scores[1]: " << scores[1];

      scores = softMax(scores);

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  vindex = " << vindex
          << ";  bbox = [" << detected_vehicles[vindex].bbox[0] << ", " << detected_vehicles[vindex].bbox[1] << ", " << detected_vehicles[vindex].bbox[2] << ", " << detected_vehicles[vindex].bbox[3] << "]"
          << ";  data[0]: " << scores[0]
          << ";  data[1]: " << scores[1];
      detected_vehicles[vindex].is_special = scores[1] > scores[0] && scores[1] > config.special_confidence;

      is_ok = true;
    }

    return is_ok;
  }

  bool Workflow::doInferenceLpdNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles)
  {
    std::vector<userver::engine::TaskWithResult<triton::client::Error>> tasks;
    tasks.reserve(detected_vehicles.size());

    std::vector<std::unique_ptr<tc::InferenceServerHttpClient>> triton_clients;
    triton_clients.resize(detected_vehicles.size());

    std::vector<tc::InferResult*> results;
    results.resize(detected_vehicles.size());

    std::vector<std::vector<uint8_t>> inputs_data;
    inputs_data.resize(detected_vehicles.size());

    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
    input_ptrs.reserve(detected_vehicles.size());

    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_ptrs;
    output_ptrs.reserve(detected_vehicles.size());

    std::vector<tc::InferOptions> options;
    options.reserve(detected_vehicles.size());

    std::vector<cv::Point2f> shifts;
    shifts.resize(detected_vehicles.size());

    std::vector<double> scales;
    scales.resize(detected_vehicles.size());

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  before inference LPDNet";

    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[vindex], config.lpd_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create inference client: $0", err.Message());
        return false;
      }
      const auto& [bbox, confidence, is_special, license_plates] = detected_vehicles[vindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  before preprocess image " << vindex << " for LPDNet";
      cv::Rect roi(cv::Point{static_cast<int>(bbox[0]), static_cast<int>(bbox[1])},
        cv::Point{static_cast<int>(bbox[2]), static_cast<int>(bbox[3])});

      // for test
      // cv::imwrite(absl::Substitute("for_lpd_net_$0_$1_$2_$3.jpg", roi.tl().x, roi.tl().y, roi.br().x, roi.br().y), img(roi));

      auto input_buffer = preprocessImageForLpdNet(img(roi), config.lpd_net_input_width, config.lpd_net_input_height, shifts[vindex],
        scales[vindex]);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  after preprocess image " << vindex << " for LPDNet";

      inputs_data[vindex].resize(config.lpd_net_input_width * config.lpd_net_input_height * 3 * sizeof(float));
      memcpy(inputs_data[vindex].data(), input_buffer.data(), inputs_data[vindex].size());
      std::vector<int64_t> shape = {1, 3, config.lpd_net_input_height, config.lpd_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.lpd_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create input data: $0", err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.lpd_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create output data: $0", err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[vindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to set up input data: $0", err.Message());
        return false;
      }
      options.emplace_back(config.lpd_net_model_name);
      options.back().model_version_ = "";

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&, vindex]
        {
          return triton_clients[vindex]->Infer(&results[vindex], options[vindex], {input_ptrs[vindex].get()}, {output_ptrs[vindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  after inference LPDNet";

    bool is_ok = false;
    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      if (auto err = tasks[vindex].Get(); !err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to send inference request (vindex = $0): $1", vindex, err.Message());
        continue;
      }

      std::shared_ptr<tc::InferResult> result_ptr(results[vindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to receive inference result (vindex = $0): $1", vindex, result_ptr->RequestStatus().Message());
        continue;
      }

      auto& [bbox, confidence, is_special, license_plates] = detected_vehicles[vindex];
      auto& detected_plates = license_plates;
      const float* data;
      size_t data_size;
      result_ptr->RawData(config.lpd_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      // the output tensor has a dimension of [14, 8400] and each column contains:
      //  0 - bbox x_center
      //  1 - bbox y_center
      //  2 - bbox width
      //  3 - bbox height
      //  4 - confidence of class 0
      //  5 - confidence of class 1
      //  6..13 - coordinates of four key points
      auto num_rows = 12 + PLATE_CLASS_COUNT;  // 12 = 4 (bbox coordinates) + 8 (key points coordinates)
      auto num_cols = 8400;
      auto bbox_index = 0;
      auto class_start_index = bbox_index + 4;
      auto kpts_start_index = num_rows - 8;
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  license plate confidence threshold (vindex = " << vindex << "): " << config.plate_confidence;
      for (auto j = 0; j < num_cols; ++j)
        for (auto k = class_start_index; k < class_start_index + PLATE_CLASS_COUNT; ++k)
          if (data[k * num_cols + j] > config.plate_confidence)
          {
            // calculating absolute coordinates of the license plate
            detected_plates.emplace_back();
            detected_plates.back().bbox[0] = bbox[0] + static_cast<float>((data[(bbox_index + 0) * num_cols + j] - data[(bbox_index + 2) * num_cols + j] / 2 - shifts[vindex].x) / scales[vindex]);
            detected_plates.back().bbox[1] = bbox[1] + static_cast<float>((data[(bbox_index + 1) * num_cols + j] - data[(bbox_index + 3) * num_cols + j] / 2 - shifts[vindex].y) / scales[vindex]);
            detected_plates.back().bbox[2] = bbox[0] + static_cast<float>((data[(bbox_index + 0) * num_cols + j] + data[(bbox_index + 2) * num_cols + j] / 2 - shifts[vindex].x) / scales[vindex]);
            detected_plates.back().bbox[3] = bbox[1] + static_cast<float>((data[(bbox_index + 1) * num_cols + j] + data[(bbox_index + 3) * num_cols + j] / 2 - shifts[vindex].y) / scales[vindex]);
            for (int l = 0; l < 8; ++l)
            {
              auto sh = shifts[vindex].x;
              auto delta = bbox[0];
              if (l % 2 == 1)
              {
                sh = shifts[vindex].y;
                delta = bbox[1];
              }
              detected_plates.back().kpts[l] = delta + static_cast<float>((data[(kpts_start_index + l) * num_cols + j] - sh) / scales[vindex]);
            }
            detected_plates.back().confidence = data[k * num_cols + j];
            detected_plates.back().plate_class = k - class_start_index;
          }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  before nms_plates count (vindex = " << vindex << "): " << detected_plates.size();
      nms_plates(detected_plates);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  after nms_plates count (vindex = " << vindex << "): " << detected_plates.size();

      // remove small detections or outside the work area
      if (!config.work_area.empty() || config.min_plate_height > 0)
        std::erase_if(detected_plates, [&config, this, img_width = img.cols, img_height = img.rows](const auto& plate)
          {
            auto do_erase = true;
            if (!config.work_area.empty())
            {
              const std::vector<cv::Point2f> plate_polygon = {
                {plate.kpts[0], plate.kpts[1]}, {plate.kpts[2], plate.kpts[3]}, {plate.kpts[4], plate.kpts[5]}, {plate.kpts[6], plate.kpts[7]}};
              std::vector<cv::Point> intersection_polygon;
              const auto plate_area = intersectConvexConvex(plate_polygon, plate_polygon, intersection_polygon, true);

              auto wa = convertToAbsolute(config.work_area, img_width, img_height);
              for (const auto& v : wa)
              {
                constexpr auto threshold = 0.999;
                intersection_polygon.clear();
                if (float intersect_area = intersectConvexConvex(v, plate_polygon, intersection_polygon, true); std::min(plate_area, intersect_area) / std::max(plate_area, intersect_area) > threshold)
                {
                  do_erase = false;
                  break;
                }
              }
            } else
              do_erase = false;

            if (do_erase && config.logs_level <= userver::logging::Level::kTrace)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                << "vstream_key = " << config.id_group << "_" << config.ext_id
                << ";  license plate does not fall completely into the work area"
                << absl::Substitute(";  key points: [($0, $1) ($2, $3) ($4, $5) ($6, $7)]", plate.kpts[0], plate.kpts[1], plate.kpts[2], plate.kpts[3], plate.kpts[4], plate.kpts[5], plate.kpts[6], plate.kpts[7]);

            // remove small license plate
            if (!do_erase && config.min_plate_height > 0
                && std::min(
                     euclidean_distance(plate.kpts[0], plate.kpts[1], plate.kpts[6], plate.kpts[7]),
                     euclidean_distance(plate.kpts[2], plate.kpts[3], plate.kpts[4], plate.kpts[5]))
                     < config.min_plate_height)
            {
              do_erase = true;
              if (config.logs_level <= userver::logging::Level::kTrace)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                  << "vstream_key = " << config.id_group << "_" << config.ext_id
                  << ";  license plate height is too small"
                  << absl::Substitute(";  key points: [($0, $1) ($2, $3) ($4, $5) ($6, $7)]", plate.kpts[0], plate.kpts[1], plate.kpts[2], plate.kpts[3], plate.kpts[4], plate.kpts[5], plate.kpts[6], plate.kpts[7]);
            }

            return do_erase; });

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  detected plate count (vindex = " << vindex << ":) " << detected_plates.size();

      is_ok = true;
    }

    return is_ok;
  }

  void Workflow::removeDuplicatePlates(const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles, int32_t width, int32_t height) const
  {
    for (int i = 0; i < static_cast<int>(detected_vehicles.size()) - 1; ++i)
      for (size_t j = i + 1; j < detected_vehicles.size(); ++j)
        if (hasIntersection(detected_vehicles[i].bbox, detected_vehicles[j].bbox))
        {
           for (auto m = detected_vehicles[i].license_plates.begin(); m != detected_vehicles[i].license_plates.end(); ++m)
             for (auto n = detected_vehicles[j].license_plates.begin(); n != detected_vehicles[j].license_plates.end(); ++n)
             {
               auto r1 = cv::Rect2f(cv::Point2f{m->bbox[0], m->bbox[1]}, cv::Point2f{m->bbox[2], m->bbox[3]});
               auto r2 = cv::Rect2f(cv::Point2f{n->bbox[0], n->bbox[1]}, cv::Point2f{n->bbox[2], n->bbox[3]});
               if (iou(r1, r2) > 0.7)  // intersection is large enough
               {
                 if (detected_vehicles[i].license_plates.size() == detected_vehicles[j].license_plates.size())
                 {
                   // remove plate from a vehicle with the largest area
                   auto b1 = cv::Rect2f(cv::Point2f{detected_vehicles[i].bbox[0], detected_vehicles[i].bbox[1]}, cv::Point2f{detected_vehicles[i].bbox[2], detected_vehicles[i].bbox[3]});
                   auto b2 = cv::Rect2f(cv::Point2f{detected_vehicles[j].bbox[0], detected_vehicles[j].bbox[1]}, cv::Point2f{detected_vehicles[j].bbox[2], detected_vehicles[j].bbox[3]});
                   if (b1.area() > b2.area())
                   {
                     if (config.logs_level <= userver::logging::Level::kTrace)
                       USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                         << "vstream_key = " << config.id_group << "_" << config.ext_id
                         << ";  remove duplicate plate number from vehicle " << i;
                     m = detected_vehicles[i].license_plates.erase(m);
                     --m;
                   } else
                   {
                     if (config.logs_level <= userver::logging::Level::kTrace)
                       USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                         << "vstream_key = " << config.id_group << "_" << config.ext_id
                         << ";  remove duplicate plate number from vehicle " << j;
                     n = detected_vehicles[j].license_plates.erase(n);
                     --n;
                   }
                 } else
                 {
                   // remove plate from a vehicle with most elements
                   if (detected_vehicles[i].license_plates.size() > detected_vehicles[j].license_plates.size())
                   {
                     if (config.logs_level <= userver::logging::Level::kTrace)
                       USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                         << "vstream_key = " << config.id_group << "_" << config.ext_id
                         << ";  remove duplicate plate number from vehicle " << i;
                     m = detected_vehicles[i].license_plates.erase(m);
                     --m;
                   } else
                   {
                     if (config.logs_level <= userver::logging::Level::kTrace)
                       USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                         << "vstream_key = " << config.id_group << "_" << config.ext_id
                         << ";  remove duplicate plate number from vehicle " << j;
                     n = detected_vehicles[j].license_plates.erase(n);
                     --n;
                   }
                 }
               }
             }
        }
    std::erase_if(detected_vehicles, [&config, img_width = width, img_height = height](const auto& vehicle)
      {
        auto do_erase = vehicle.license_plates.size() == 0 && !vehicle.is_special;
        if (!config.work_area.empty() && !do_erase)
        {
          bool has_intersection = false;
          const std::vector<cv::Point2f> vehicle_polygon = {
            {vehicle.bbox[0], vehicle.bbox[1]},
            {vehicle.bbox[2], vehicle.bbox[1]},
            {vehicle.bbox[2], vehicle.bbox[3]},
            {vehicle.bbox[0], vehicle.bbox[3]}};
          auto wa = convertToAbsolute(config.work_area, img_width, img_height);
          for (const auto& v : wa)
          {
            std::vector<cv::Point> intersection_polygon;
            intersectConvexConvex(v, vehicle_polygon, intersection_polygon, true);
            if (intersection_polygon.size() >= 3)
            {
              has_intersection = true;
              break;
            }
          }
          do_erase = !has_intersection;
        }
        return do_erase;
      });
  }

  bool Workflow::isValidPlateNumber(const absl::string_view plate_number, const int32_t plate_class)
  {
    if (plate_class == PLATE_CLASS_RU_1 || plate_class == PLATE_CLASS_RU_1A)
    {
      const HashSet<char> numbers = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
      const HashSet<char> letters = {'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y'};

      if (plate_number.size() < 8 || plate_number.size() > 9)
        return false;

      for (size_t i = 0; i < plate_number.size(); ++i)
        if (i == 0 || i == 4 || i == 5)
        {
          if (!letters.contains(plate_number[i]))
            return false;
        } else
        {
          if (!numbers.contains(plate_number[i]))
            return false;
        }
    }

    return true;
  }

  bool Workflow::doInferenceLprNet(const cv::Mat& img, const VStreamConfig& config, std::vector<LicensePlate*>& detected_plates)
  {
    std::vector labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
      "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ""};

    // scale values by plate class to calculate height of an image after perspective transformation
    std::vector scale_height_by_plate_class = {
      112.0f / 520.f,   // 0 - Russian type 1
      170.0f / 290.0f,  // 1 - Russian type 1A
    };

    std::vector<userver::engine::TaskWithResult<triton::client::Error>> tasks;
    tasks.reserve(detected_plates.size());

    std::vector<std::unique_ptr<tc::InferenceServerHttpClient>> triton_clients;
    triton_clients.resize(detected_plates.size());

    std::vector<tc::InferResult*> results;
    results.resize(detected_plates.size());

    std::vector<std::vector<uint8_t>> inputs_data;
    inputs_data.resize(detected_plates.size());

    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
    input_ptrs.reserve(detected_plates.size());

    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_ptrs;
    output_ptrs.reserve(detected_plates.size());

    std::vector<tc::InferOptions> options;
    options.reserve(detected_plates.size());

    std::vector<cv::Point2f> shifts;
    shifts.resize(detected_plates.size());

    std::vector<double> scales;
    scales.resize(detected_plates.size());

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  before inference LPRNet";

    for (size_t pindex = 0; pindex < detected_plates.size(); ++pindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[pindex], config.lpr_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create inference client: $0", err.Message());
        return false;
      }
      auto& [bbox, confidence, kpts, plate_class, plate_numbers] = *detected_plates[pindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  before preprocess image " << pindex << " for LPRNet";

      auto src = cv::Mat(4, 2, CV_32F, kpts);
      std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(config.lpr_net_input_width - 1), 0.0f},
        {static_cast<float>(config.lpr_net_input_width - 1), static_cast<float>(config.lpr_net_input_width) * scale_height_by_plate_class[plate_class] - 1},
        {0.0f, static_cast<float>(config.lpr_net_input_width) * scale_height_by_plate_class[plate_class] - 1}};
      auto transform_mat = cv::getPerspectiveTransform(src, dst);
      cv::Mat lp_image;
      cv::warpPerspective(img, lp_image, transform_mat, {config.lpr_net_input_width, static_cast<int>(static_cast<float>(config.lpr_net_input_width) * scale_height_by_plate_class[plate_class])});

      // for test
      // cv::imwrite(absl::Substitute("pp_$0.jpg", pindex), lp_image);

      auto input_buffer = preprocessImageForLprNet(lp_image, config.lpr_net_input_width, config.lpr_net_input_height, shifts[pindex],
        scales[pindex]);

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  after preprocess image " << pindex << " for LPRNet";

      inputs_data[pindex].resize(config.lpr_net_input_width * config.lpr_net_input_height * 3 * sizeof(float));
      memcpy(inputs_data[pindex].data(), input_buffer.data(), inputs_data[pindex].size());
      std::vector<int64_t> shape = {1, 3, config.lpr_net_input_height, config.lpr_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.lpr_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create input data: $0", err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.lpr_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to create output data: $0", err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[pindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to set up input data: $0", err.Message());
        return false;
      }

      options.emplace_back(config.lpr_net_model_name);
      options.back().model_version_ = "";

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&, pindex]
        {
          return triton_clients[pindex]->Infer(&results[pindex], options[pindex], {input_ptrs[pindex].get()}, {output_ptrs[pindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
        << "vstream_key = " << config.id_group << "_" << config.ext_id
        << ";  after inference LPRNet";

    bool is_ok = false;
    for (size_t pindex = 0; pindex < detected_plates.size(); ++pindex)
    {
      auto err = tasks[pindex].Get();
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to send inference request (vindex = $0): $1", pindex, err.Message());
        continue;
      }

      std::shared_ptr<tc::InferResult> result_ptr(results[pindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_) << absl::Substitute("Error! Unable to receive inference result (vindex = $0): $1", pindex, err.Message());
        continue;
      }

      auto& plate = *detected_plates[pindex];
      const float* data;
      size_t data_size;
      result_ptr->RawData(config.lpr_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      std::vector<CharData> chars_data;
      auto num_rows = 40;
      auto num_cols = 525;
      auto bbox_index = 0;
      auto class_start_index = bbox_index + 4;
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  char score threshold: " << config.char_score;
      for (auto j = 0; j < num_cols; ++j)
        for (auto k = class_start_index; k < num_rows; ++k)
          if (data[k * num_cols + j] > config.char_score)
          {
            chars_data.emplace_back();
            chars_data.back().bbox[0] = static_cast<float>(
              (data[(bbox_index + 0) * num_cols + j] - data[(bbox_index + 2) * num_cols + j] / 2 - shifts[pindex].x) / scales[pindex]);
            chars_data.back().bbox[1] = static_cast<float>(
              (data[(bbox_index + 1) * num_cols + j] - data[(bbox_index + 3) * num_cols + j] / 2 - shifts[pindex].y) / scales[pindex]);
            chars_data.back().bbox[2] = static_cast<float>(
              (data[(bbox_index + 0) * num_cols + j] + data[(bbox_index + 2) * num_cols + j] / 2 - shifts[pindex].x) / scales[pindex]);
            chars_data.back().bbox[3] = static_cast<float>(
              (data[(bbox_index + 1) * num_cols + j] + data[(bbox_index + 3) * num_cols + j] / 2 - shifts[pindex].y) / scales[pindex]);

            chars_data.back().confidence = data[k * num_cols + j];
            chars_data.back().char_class = k - class_start_index;
            chars_data.back().plate_class = plate.plate_class;
          }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  before nms_chars count (pindex = " << pindex << "): " << chars_data.size();
      nms_chars(chars_data, config.char_iou_threshold);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
          << "vstream_key = " << config.id_group << "_" << config.ext_id
          << ";  after nms_chars count (pindex = " << pindex << "): " << chars_data.size();

      // assembling license plate numbers from char data
      std::ranges::sort(chars_data, cmp_chars_position);
      HashSet<size_t> used_indices;
      plate.plate_numbers.push_back({"", 1.0f});
      for (size_t i = 0; i < chars_data.size(); ++i)
        if (!used_indices.contains(i))
        {
          std::vector<size_t> new_char_indices;
          new_char_indices.push_back(i);
          for (size_t j = i + 1; j < chars_data.size(); ++j)
            if (!used_indices.contains(j))
            {
              auto r1 = cv::Rect2f(cv::Point2f{chars_data[i].bbox[0], chars_data[i].bbox[1]}, cv::Point2f{chars_data[i].bbox[2], chars_data[i].bbox[3]});
              if (auto r2 = cv::Rect2f(cv::Point2f{chars_data[j].bbox[0], chars_data[j].bbox[1]}, cv::Point2f{chars_data[j].bbox[2], chars_data[j].bbox[3]}); iou(r1, r2) > config.char_iou_threshold)
              {
                new_char_indices.push_back(j);
                used_indices.insert(j);
              }
            }

          if (new_char_indices.size() > 1)
          {
            auto copy_data = plate.plate_numbers;
            for (size_t k = 1; k < new_char_indices.size(); ++k)
              plate.plate_numbers.insert(plate.plate_numbers.end(), copy_data.begin(), copy_data.end());
          }
          for (size_t k = 0; k < plate.plate_numbers.size(); ++k)
          {
            auto m = k * new_char_indices.size() / plate.plate_numbers.size();
            plate.plate_numbers[k].number += labels[chars_data[new_char_indices[m]].char_class];
            plate.plate_numbers[k].score *= chars_data[new_char_indices[m]].confidence;
          }
        }

      // remove invalid numbers
      std::erase_if(plate.plate_numbers,
        [this, &plate, &config](const auto& item)
        {
          auto is_valid = isValidPlateNumber(item.number, plate.plate_class);
          if (!is_valid)
            if (config.logs_level <= userver::logging::Level::kTrace)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace)
                << "vstream_key = " << config.id_group << "_" << config.ext_id
                << ";  invalid plate number: " << item.number;
          return !is_valid;
        });

      // sort results in descending order by score
      std::ranges::sort(plate.plate_numbers,
        [](const auto& left, const auto& right)
        {
          return left.score > right.score;
        });

      is_ok = true;
    }

    return is_ok;
  }

  int64_t Workflow::addEventLog(const int32_t id_vstream, const userver::storages::postgres::TimePointTz& log_date, const userver::formats::json::Value& info) const
  {
    const userver::storages::postgres::Query query{SQL_ADD_EVENT};
    int64_t result = -1;
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      if (const auto res = trx.Execute(query, id_vstream, log_date, info); !res.IsEmpty())
        result = res.AsSingleRow<int64_t>();
      trx.Commit();
    } catch (...)
    {
      trx.Rollback();
    }

    return result;
  }
}  // namespace Lprs
