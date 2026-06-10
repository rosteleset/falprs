#include <algorithm>
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
#include "image_preprocessing.hpp"

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
        if (plate_class == dets[n].plate_class_common && hasIntersection(bbox, dets[n].bbox))
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
    if (a.plate_class == PLATE_CLASS_RU_1 || a.plate_class == PLATE_CLASS_BY || a.plate_class == PLATE_CLASS_AM)
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

  inline std::vector<std::vector<cv::Point>> convertToAbsolute(const std::vector<std::vector<cv::Point2f>>& work_area, const int32_t width, const int32_t height)
  {
    std::vector<std::vector<cv::Point>> wa(work_area.size());
    for (size_t i = 0; i < work_area.size(); ++i)
    {
      wa[i].reserve(work_area[i].size());
      for (size_t j = 0; j < work_area[i].size(); ++j)
        wa[i].emplace_back(static_cast<int>(work_area[i][j].x * static_cast<float>(width) / 100.0f),
          static_cast<int>(work_area[i][j].y * static_cast<float>(height) / 100.0f));
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
    local_config_.allow_group_id_without_auth = config[ConfigParams::SECTION_NAME][ConfigParams::ALLOW_GROUP_ID_WITHOUT_AUTH].As<decltype(local_config_.allow_group_id_without_auth)>(local_config_.allow_group_id_without_auth);
    local_config_.ban_maintenance_interval = config[ConfigParams::SECTION_NAME][ConfigParams::BAN_MAINTENANCE_INTERVAL].As<decltype(local_config_.ban_maintenance_interval)>(local_config_.ban_maintenance_interval);
    local_config_.events_log_maintenance_interval = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_LOG_MAINTENANCE_INTERVAL].As<decltype(local_config_.events_log_maintenance_interval)>(local_config_.events_log_maintenance_interval);
    local_config_.events_log_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_LOG_TTL].As<decltype(local_config_.events_log_ttl)>(local_config_.events_log_ttl);
    local_config_.events_screenshots_path = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_SCREENSHOTS_PATH].As<decltype(local_config_.events_screenshots_path)>(local_config_.events_screenshots_path);
    if (!local_config_.events_screenshots_path.empty() && !local_config_.events_screenshots_path.ends_with("/"))
      local_config_.events_screenshots_path += "/";
    local_config_.events_screenshots_url_prefix = config[ConfigParams::SECTION_NAME][ConfigParams::EVENTS_SCREENSHOTS_URL_PREFIX].As<decltype(local_config_.events_screenshots_url_prefix)>(local_config_.events_screenshots_url_prefix);
    if (!local_config_.events_screenshots_url_prefix.empty() && !local_config_.events_screenshots_url_prefix.ends_with("/"))
      local_config_.events_screenshots_url_prefix += "/";
    local_config_.failed_path = config[ConfigParams::SECTION_NAME][ConfigParams::FAILED_PATH].As<decltype(local_config_.failed_path)>(local_config_.failed_path);
    if (!local_config_.failed_path.empty() && !local_config_.failed_path.ends_with("/"))
      local_config_.failed_path += "/";
    local_config_.failed_ttl = config[ConfigParams::SECTION_NAME][ConfigParams::FAILED_TTL].As<decltype(local_config_.failed_ttl)>(local_config_.failed_ttl);

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
    {
      auto frame_url = config.screenshot_url.starts_with("data:") ? "data:base64..." : config.screenshot_url;
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "Start processPipeline: vstream_key = {};  frame_url = {}",
        vstream_key, frame_url);
    }

    try
    {
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  before image acquisition",
          vstream_key);

      std::string image_data;
      if (config.screenshot_url.starts_with("data:"))
      {
        if (auto pos_comma = config.screenshot_url.find(','); pos_comma != std::string::npos)
          if (config.screenshot_url.find(";base64,") != std::string::npos)
            if (!absl::Base64Unescape(absl::ClippedSubstr(config.screenshot_url, pos_comma + 1), &image_data))
            {
              if (config.logs_level <= userver::logging::Level::kError)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
                  "Error decoding image from BASE64: vstream_key = {};",
                  vstream_key);

              return;
            }
      } else
      {
        // parse user and password
        std::string auth_user;
        std::string auth_password;
        if (auto char_alpha = config.screenshot_url.find('@'); char_alpha != std::string::npos)
        {
          if (auto protocol_suffix = config.screenshot_url.find("://"); protocol_suffix != std::string::npos && protocol_suffix < char_alpha)
          {
            auto start_of_authority = protocol_suffix + 3;
            if (auto char_colon = config.screenshot_url.find(':', start_of_authority); char_colon != std::string::npos && char_colon < char_alpha)
            {
              auth_user = config.screenshot_url.substr(start_of_authority, char_colon - start_of_authority);
              auth_password = config.screenshot_url.substr(char_colon + 1, char_alpha - char_colon - 1);
            } else
              auth_user = config.screenshot_url.substr(start_of_authority, char_alpha - start_of_authority);
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

        if (capture_response->status_code() != userver::clients::http::Status::OK || capture_response->body_view().empty())
        {
          if (config.logs_level <= userver::logging::Level::kError)
            USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
              "vstream_key = {};  url = {};  status_code = {}",
              vstream_key, config.screenshot_url, capture_response->status_code());
          if (config.delay_after_error.count() > 0)
          {
            if (config.logs_level <= userver::logging::Level::kError)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
                "vstream_key = {};  delay for {}ms",
                vstream_key, config.delay_after_error.count());
            nextPipeline(std::move(vstream_key), config.delay_after_error);
          } else
            stopWorkflow(std::move(vstream_key));

          return;
        }

        image_data = capture_response->body();
      }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  after image acquisition",
          vstream_key);

      if (config.logs_level <= userver::logging::Level::kTrace)
      {
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  image size = {} bytes",
          vstream_key, image_data.size());
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  before decoding the image",
          vstream_key);
      }
      cv::Mat frame = imdecode(std::vector(image_data.begin(), image_data.end()),
        cv::IMREAD_COLOR);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  after decoding the image",
          vstream_key);

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
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  before doInferenceVdNet",
          vstream_key);
      doInferenceVdNet(frame, config, detected_vehicles);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  after doInferenceVdNet",
          vstream_key);

      if (config.flag_process_special)
      {
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  before doInferenceVcNet",
            vstream_key);
        doInferenceVcNet(frame, config, detected_vehicles);
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  after doInferenceVcNet",
            vstream_key);
      }

      if (config.logs_level <= userver::logging::Level::kTrace)
        for (size_t i = 0; i < detected_vehicles.size(); ++i)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  vehicle {} confidence: {:.3f}",
            vstream_key, i, detected_vehicles[i].confidence);

      // check for a special vehicles ban
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
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  special vehicles are banned ({}s left)",
                vstream_key, std::chrono::duration_cast<std::chrono::seconds>((*ban_special_data_ptr)[vstream_key] - now).count());
          } else
          {
            ban_special_data_ptr->erase(vstream_key);
            if (config.logs_level <= userver::logging::Level::kTrace)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {};  special vehicles are no longer banned",
                vstream_key);
          }
        }
      }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  before doInferenceLpdNet",
          vstream_key);
      doInferenceLpdNet(frame, config, detected_vehicles);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {};  after doInferenceLpdNet",
          vstream_key);

      removeDuplicatePlates(config, detected_vehicles, frame.cols, frame.rows);

      // for test
      // save images of the special vehicles
      /*for (size_t v = 0; v < detected_vehicles.size(); ++v)
        if (detected_vehicles[v].is_special)
        {
          cv::Rect roi(cv::Point{static_cast<int>(detected_vehicles[v].bbox[0]), static_cast<int>(detected_vehicles[v].bbox[1])},
            cv::Point{static_cast<int>(detected_vehicles[v].bbox[2]), static_cast<int>(detected_vehicles[v].bbox[3])});
          imwrite(absl::Substitute("special_$0_$1.jpg", config.id_vstream, v), frame(roi));
        }*/

      std::vector<LicensePlate*> detected_plates;
      for (const auto& [bbox, confidence, is_special, license_plates] : detected_vehicles)
        for (const auto& license_plate : license_plates)
          detected_plates.push_back(const_cast<LicensePlate*>(&license_plate));

      bool result = true;
      if (!detected_plates.empty())
      {
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  before doInferenceLpcNet",
            vstream_key);
        result = doInferenceLpcNet(frame, config, detected_plates);
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  after doInferenceLpcNet",
            vstream_key);

        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  before doInferenceLprNet",
            vstream_key);
        result = doInferenceLprNet(frame, config, detected_plates);
        if (config.logs_level <= userver::logging::Level::kTrace)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
            "vstream_key = {};  after doInferenceLprNet",
            vstream_key);
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
          for (const auto& [bbox_plate, confidence_plate, kpts, plate_class_common, plate_numbers] : license_plates)
          {
            has_failed = has_failed || plate_numbers.empty();
            for (const auto& [plate_class, number, score] : plate_numbers)
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
                auto banned_bbox = cv::Rect2f(cv::Point2f{bbox_plate[0], bbox_plate[1]}, cv::Point2f{bbox_plate[2], bbox_plate[3]});
                auto data_ptr = ban_data.Lock();
                if (data_ptr->contains(k))
                {
                  is_banned = (*data_ptr)[k].tp1 > now;
                  if (is_banned)
                  {
                    if (config.logs_level <= userver::logging::Level::kDebug)
                      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
                        "vstream_key = {};  plate number {} is banned at the first stage ({}s left)",
                        vstream_key, number, std::chrono::duration_cast<std::chrono::seconds>((*data_ptr)[k].tp1 - now).count());
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
                        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
                          "vstream_key = {};  plate_number {} is banned at the second stage (iou = {:.2f}, threshold = {:.2f})",
                          vstream_key, number, iou_value, config.ban_iou_threshold);
                      else
                        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
                          "vstream_key = {};  plate_numbers {} was removed from the the second stage ban (iou = {:.2f}, threshold = {:.2f})",
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
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kInfo,
                  "vstream_key = {};  plate number: {}",
                  vstream_key, number);
              userver::formats::json::ValueBuilder plate_data;
              plate_data[Api::PARAM_BOX] = userver::formats::json::MakeArray(
                static_cast<int32_t>(bbox_plate[0]),
                static_cast<int32_t>(bbox_plate[1]),
                static_cast<int32_t>(bbox_plate[2]),
                static_cast<int32_t>(bbox_plate[3]));
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

          // write a screenshot to a file
          auto path_prefix = absl::StrCat(local_config_.events_screenshots_path, path_suffix);
          userver::fs::CreateDirectories(fs_task_processor_, path_prefix);
          auto path = absl::StrCat(path_prefix, uuid, screenshot_extension);
          userver::fs::RewriteFileContents(fs_task_processor_, path, image_data);
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
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kWarning,
                  "vstream_key = {};  error sending data to callback {}",
                  vstream_key, config.callback_url);
          } catch (std::exception& e)
          {
            LOG_ERROR_TO(logger_,
              "vstream_key = {};  error sending data to callback {};  {}",
              vstream_key, config.callback_url, e.what());
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

          // write a failed screenshot to a file
          auto path_prefix = absl::StrCat(local_config_.failed_path, path_suffix);
          userver::fs::CreateDirectories(fs_task_processor_, path_prefix);
          auto path = absl::StrCat(path_prefix, uuid, screenshot_extension);
          auto path_draw = absl::StrCat(path_prefix, uuid, "_draw", screenshot_extension);
          userver::fs::RewriteFileContents(fs_task_processor_, path, image_data);
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
            auto& [bbox, confidence, kpts, plate_class_common, plate_numbers] = *plate_ptr;
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
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
          "vstream_key = {};  {}",
          vstream_key, e.what());

      if (config.delay_after_error.count() > 0)
      {
        if (config.logs_level <= userver::logging::Level::kError)
          USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kError,
            "vstream_key = {};  delay for {}ms",
            vstream_key, config.delay_after_error.count());
        nextPipeline(std::move(vstream_key), config.delay_after_error);
      } else
        stopWorkflow(std::move(vstream_key));

      return;
    }

    if (config.logs_level <= userver::logging::Level::kDebug)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "End processPipeline: vstream_key = {};",
        vstream_key);

    nextPipeline(std::move(vstream_key), config.delay_between_frames);
  }

  void Workflow::doBanMaintenance()
  {
    LOG_DEBUG_TO(logger_, "call doBanMaintenance");
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
    LOG_DEBUG_TO(logger_, "call doEventsLogMaintenance");
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
      LOG_INFO_TO(logger_,
        "Stopping a workflow by timeout: vstream_key = {};",
        vstream_key);

    if (do_next)
      tasks_.Detach(AsyncNoSpan(task_processor_, &Workflow::processPipeline, this, std::move(vstream_key)));
  }

  // Inference pipeline methods
  bool Workflow::doInferenceVdNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles) const
  {
    detected_vehicles.clear();

    std::unique_ptr<tc::InferenceServerHttpClient> triton_client;
    auto err = tc::InferenceServerHttpClient::Create(&triton_client, config.vd_net_inference_server, false);
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_,
        "Error! Unable to create inference client: {}",
        err.Message());
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before preprocess image for VDNet",
        config.id_group, config.ext_id);
    cv::Point2f shift;
    double scale;
    auto blob = prepareBlobForYOLO(img, config.lpd_net_input_width, config.lpd_net_input_height, shift,
      scale);
    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after preprocess image for VDNet",
        config.id_group, config.ext_id);

    const auto* raw = blob.ptr<uint8_t>();
    const auto byte_size = blob.total() * blob.elemSize();
    std::vector input_data(raw, raw + byte_size);
    std::vector<int64_t> shape = {1, 3, config.vd_net_input_height, config.vd_net_input_width};
    tc::InferInput* input;
    err = tc::InferInput::Create(&input, config.vd_net_input_tensor_name, shape, "FP32");
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_,
        "Error! Unable to create input data: {}",
        err.Message());
      return false;
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);

    tc::InferRequestedOutput* output;
    err = tc::InferRequestedOutput::Create(&output, config.vd_net_output_tensor_name);
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_,
        "Error! Unable to create output data: {}",
        err.Message());
      return false;
    }
    std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

    std::vector inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};
    err = input_ptr->AppendRaw(input_data);
    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_,
        "Error! Unable to set up input data: {}",
        err.Message());
      return false;
    }

    tc::InferOptions options(config.vd_net_model_name);
    options.model_version_ = "";
    // inference timeout in microseconds
    options.client_timeout_ = std::chrono::duration_cast<std::chrono::microseconds>(config.inference_timeout).count();
    tc::InferResult* result;

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before inference VDNet",
        config.id_group, config.ext_id);

    AsyncNoSpan(fs_task_processor_,
      [&err, &triton_client, &result, &options, &inputs, &outputs]
      {
        err = triton_client->Infer(&result, options, inputs, outputs);
      }).Get();

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after inference VDNet",
        config.id_group, config.ext_id);

    if (!err.IsOk())
    {
      LOG_ERROR_TO(logger_,
        "Error! Unable to send inference request: {}",
        err.Message());
      return false;
    }

    std::shared_ptr<tc::InferResult> result_ptr(result);
    if (!result_ptr->RequestStatus().IsOk())
    {
      LOG_ERROR_TO(logger_,
        "Error! Unable to receive inference result: {}",
        err.Message());
      return false;
    }

    if (config.logs_level <= userver::logging::Level::kDebug)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kDebug,
        "vstream_key = {}_{};  inference VDNet OK",
        config.id_group, config.ext_id);

    const float* data;
    size_t data_size;
    result_ptr->RawData(config.vd_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

    // the output tensor has a dimension of [5, 8400]
    //  0 - bbox x_center
    //  1 - bbox y_center
    //  2 - bbox width
    //  3 - bbox height
    //  4 - confidence
    auto num_cols = 8400;
    auto bbox_index = 0;
    auto class_start_index = bbox_index + 4;
    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  vehicle confidence threshold: {:.3f}",
        config.id_group, config.ext_id, config.vehicle_confidence);
    for (auto j = 0; j < num_cols; ++j)
    {
      const float cx = data[(bbox_index + 0) * num_cols + j];
      const float cy = data[(bbox_index + 1) * num_cols + j];
      const float bw = data[(bbox_index + 2) * num_cols + j];
      const float bh = data[(bbox_index + 3) * num_cols + j];
      auto k = class_start_index;
      if (auto conf = data[k * num_cols + j]; conf > config.vehicle_confidence)
      {
        const float inv_scale = 1.0f / static_cast<float>(scale);
        auto x_min = std::fmax((cx - bw / 2 - shift.x) * inv_scale, 0.0f);
        auto y_min = std::fmax((cy - bh / 2 - shift.y) * inv_scale, 0.0f);
        auto x_max = std::fmin((cx + bw / 2 - shift.x) * inv_scale, static_cast<float>(img.cols - 1));
        auto y_max = std::fmin((cy + bh / 2 - shift.y) * inv_scale, static_cast<float>(img.rows - 1));

        // remove small vehicle detections
        auto vehicle_area = (x_max - x_min + 1) * (y_max - y_min + 1);
        if (auto screen_area = static_cast<float>(img.cols * img.rows); vehicle_area / screen_area < config.vehicle_area_ratio_threshold)
          continue;

        detected_vehicles.emplace_back();
        auto& vehicle = detected_vehicles.back();
        vehicle.bbox[0] = x_min;
        vehicle.bbox[1] = y_min;
        vehicle.bbox[2] = x_max;
        vehicle.bbox[3] = y_max;
        vehicle.confidence = conf;
      }
    }

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before nms_vehicles count: {}",
        config.id_group, config.ext_id, detected_vehicles.size());
    nms_vehicles(detected_vehicles, config.vehicle_iou_threshold);
    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after nms_vehicles count: {}",
        config.id_group, config.ext_id, detected_vehicles.size());

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
    inputs_data.reserve(detected_vehicles.size());

    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
    input_ptrs.reserve(detected_vehicles.size());

    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_ptrs;
    output_ptrs.reserve(detected_vehicles.size());

    std::vector<tc::InferOptions> options;
    options.reserve(detected_vehicles.size());

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before inference VcNet",
        config.id_group, config.ext_id);

    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[vindex], config.vc_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create inference client: {}",
          err.Message());
        return false;
      }

      const auto& [bbox, confidence, is_special, license_plates] = detected_vehicles[vindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  before preprocess image {} for VcNet",
          config.id_group, config.ext_id, vindex);
      cv::Rect roi(cv::Point{static_cast<int>(bbox[0]), static_cast<int>(bbox[1])},
        cv::Point{static_cast<int>(bbox[2]), static_cast<int>(bbox[3])});
      auto blob = prepareBlobForVcNet(img(roi), config.vc_net_input_width, config.vc_net_input_height);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  after preprocess image {} for VcNet",
          config.id_group, config.ext_id, vindex);

      const auto* raw = blob.ptr<uint8_t>();
      const auto byte_size = blob.total() * blob.elemSize();
      inputs_data.emplace_back(raw, raw + byte_size);
      std::vector<int64_t> shape = {1, 3, config.vc_net_input_height, config.vc_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.vc_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create input data: {}",
          err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.vc_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create output data: {}",
          err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[vindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to set up input data: {}",
          err.Message());
        return false;
      }
      options.emplace_back(config.vc_net_model_name);
      options.back().model_version_ = "";
      // inference timeout in microseconds
      options.back().client_timeout_ = std::chrono::duration_cast<std::chrono::microseconds>(config.inference_timeout).count();

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&triton_clients, &results, &options, &input_ptrs, &output_ptrs, vindex]
        {
          return triton_clients[vindex]->Infer(&results[vindex], options[vindex], {input_ptrs[vindex].get()}, {output_ptrs[vindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after inference VcNet",
        config.id_group, config.ext_id);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  special confidence threshold: {:.3f}",
        config.id_group, config.ext_id, config.special_confidence);

    bool is_ok = false;
    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      if (auto err = tasks[vindex].Get(); !err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to send inference request (vindex = {}): {}",
          vindex, err.Message());
        continue;
      }

      const std::shared_ptr<tc::InferResult> result_ptr(results[vindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to receive inference result (vindex = {}): {}",
          vindex, result_ptr->RequestStatus().Message());
        continue;
      }

      const float* data;
      size_t data_size;
      result_ptr->RawData(config.vc_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      std::vector<float> scores;
      scores.assign(data, data + static_cast<int>(data_size / sizeof(float)));

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  vindex = {};  softmax scores: ",
          config.id_group, config.ext_id, vindex) << scores;

      scores = softMax(scores);

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  vindex = {};  bbox = [{:.2f}, {:.2f}, {:.2f}, {:.2f}];  scores: ",
          config.id_group, config.ext_id, vindex,
          detected_vehicles[vindex].bbox[0], detected_vehicles[vindex].bbox[1], detected_vehicles[vindex].bbox[2], detected_vehicles[vindex].bbox[3]) << scores;
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
    inputs_data.reserve(detected_vehicles.size());

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
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before inference LPDNet",
        config.id_group, config.ext_id);

    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[vindex], config.lpd_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create inference client: {}",
          err.Message());
        return false;
      }
      const auto& [bbox, confidence, is_special, license_plates] = detected_vehicles[vindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  before preprocess image {} for LPDNet",
          config.id_group, config.ext_id, vindex);
      cv::Rect roi(cv::Point{static_cast<int>(bbox[0]), static_cast<int>(bbox[1])},
        cv::Point{static_cast<int>(bbox[2]), static_cast<int>(bbox[3])});

      // for test
      // cv::imwrite(absl::Substitute("for_lpd_net_$4_$5_$0_$1_$2_$3.jpg", roi.tl().x, roi.tl().y, roi.br().x, roi.br().y, config.ext_id, vindex), img(roi));

      auto blob = prepareBlobForYOLO(img(roi), config.lpd_net_input_width, config.lpd_net_input_height, shifts[vindex],
        scales[vindex]);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  after preprocess image {} for LPDNet",
          config.id_group, config.ext_id, vindex);

      const auto* raw = blob.ptr<uint8_t>();
      const auto byte_size = blob.total() * blob.elemSize();
      inputs_data.emplace_back(raw, raw + byte_size);
      std::vector<int64_t> shape = {1, 3, config.lpd_net_input_height, config.lpd_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.lpd_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create input data: {}",
          err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.lpd_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create output data: {}",
          err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[vindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to set up input data: {}",
          err.Message());
        return false;
      }
      options.emplace_back(config.lpd_net_model_name);
      options.back().model_version_ = "";
      // inference timeout in microseconds
      options.back().client_timeout_ = std::chrono::duration_cast<std::chrono::microseconds>(config.inference_timeout).count();

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&triton_clients, &results, &options, &input_ptrs, &output_ptrs, vindex]
        {
          return triton_clients[vindex]->Infer(&results[vindex], options[vindex], {input_ptrs[vindex].get()}, {output_ptrs[vindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after inference LPDNet",
        config.id_group, config.ext_id);

    bool is_ok = false;
    for (size_t vindex = 0; vindex < detected_vehicles.size(); ++vindex)
    {
      if (auto err = tasks[vindex].Get(); !err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to send inference request (vindex = {}): {}",
          vindex, err.Message());
        continue;
      }

      std::shared_ptr<tc::InferResult> result_ptr(results[vindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to receive inference result (vindex = {}): {}",
          vindex, result_ptr->RequestStatus().Message());
        continue;
      }

      auto& [bbox, confidence, is_special, license_plates] = detected_vehicles[vindex];
      auto& detected_plates = license_plates;
      const float* data;
      size_t data_size;
      result_ptr->RawData(config.lpd_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      // the output tensor has a dimension of [300, 14], and each row contains:
      //  0 - bbox left
      //  1 - bbox top
      //  2 - bbox right
      //  3 - bbox bottom
      //  4 - confidence
      //  5 - class
      //  6..13 - coordinates of four key points clockwise starting from the left upper corner

      auto num_cols = 14;
      auto num_rows = data_size / num_cols / sizeof(float);
      auto bbox_index = 0;
      auto conf_index = bbox_index + 4;
      auto kpts_start_index = num_cols - 8;
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  license plate confidence threshold (vindex = {}): {}",
          config.id_group, config.ext_id, vindex, config.plate_confidence);
      for (size_t i = 0; i < num_rows; ++i)
        if (const float conf = data[conf_index + num_cols * i]; conf > config.plate_confidence)
        {
          // for test
          // std::cout << "conf: " << conf << std::endl;
          // calculating absolute coordinates of the license plate
          detected_plates.emplace_back();
          auto& plate = detected_plates.back();
          const float left = data[bbox_index + 0 + num_cols * i];
          const float top = data[bbox_index + 1 + num_cols * i];
          const float right = data[bbox_index + 2 + num_cols * i];
          const float bottom = data[bbox_index + 3 + num_cols * i];
          const float inv_scale = 1.0f / static_cast<float>(scales[vindex]);
          plate.bbox[0] = bbox[0] + (left - shifts[vindex].x) * inv_scale;
          plate.bbox[1] = bbox[1] + (top - shifts[vindex].y) * inv_scale;
          plate.bbox[2] = bbox[0] + (right - shifts[vindex].x) * inv_scale;
          plate.bbox[3] = bbox[1] + (bottom - shifts[vindex].y) * inv_scale;
          for (int l = 0; l < 8; ++l)
          {
            auto sh = shifts[vindex].x;
            auto delta = bbox[0];
            if (l % 2 == 1)
            {
              sh = shifts[vindex].y;
              delta = bbox[1];
            }
            detected_plates.back().kpts[l] = delta + static_cast<float>((data[kpts_start_index + l + num_cols * i] - sh) / scales[vindex]);
          }
          plate.confidence = data[conf_index + num_cols * i];
          plate.plate_class_common = -1;
        }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  before nms_plates count (vindex = {}): {}",
          config.id_group, config.ext_id, vindex, detected_plates.size());
      nms_plates(detected_plates);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  after nms_plates count (vindex = {}): {}",
          config.id_group, config.ext_id, vindex, detected_plates.size());

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

              for (const auto wa = convertToAbsolute(config.work_area, img_width, img_height); const auto& v : wa)
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
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {}_{};  license plate does not fall completely into the work area;  key points: [({:.2f}, {:.2f}) ({:.2f}, {:.2f}) ({:.2f}, {:.2f}) ({:.2f}, {:.2f})]",
                config.id_group, config.ext_id,
                plate.kpts[0], plate.kpts[1], plate.kpts[2], plate.kpts[3], plate.kpts[4], plate.kpts[5], plate.kpts[6], plate.kpts[7]);

            // remove small license plate
            if (!do_erase && config.min_plate_height > 0
                && std::min(
                     euclidean_distance(plate.kpts[0], plate.kpts[1], plate.kpts[6], plate.kpts[7]),
                     euclidean_distance(plate.kpts[2], plate.kpts[3], plate.kpts[4], plate.kpts[5]))
                     < config.min_plate_height)
            {
              do_erase = true;
              if (config.logs_level <= userver::logging::Level::kTrace)
                USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                  "vstream_key = {}_{};  license plate height is too small;  key points: [({:.2f}, {:.2f}) ({:.2f}, {:.2f}) ({:.2f}, {:.2f}) ({:.2f}, {:.2f})]",
                  config.id_group, config.ext_id,
                  plate.kpts[0], plate.kpts[1], plate.kpts[2], plate.kpts[3], plate.kpts[4], plate.kpts[5], plate.kpts[6], plate.kpts[7]);
            }

            return do_erase; });

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  detected plate count (vindex = {}): {}",
          config.id_group, config.ext_id, vindex, detected_plates.size());

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
              if (iou(r1, r2) > 0.7)  // the intersection is large enough
              {
                if (detected_vehicles[i].license_plates.size() == detected_vehicles[j].license_plates.size())
                {
                  // remove plate from a vehicle with the largest area
                  auto b1 = cv::Rect2f(cv::Point2f{detected_vehicles[i].bbox[0], detected_vehicles[i].bbox[1]}, cv::Point2f{detected_vehicles[i].bbox[2], detected_vehicles[i].bbox[3]});
                  auto b2 = cv::Rect2f(cv::Point2f{detected_vehicles[j].bbox[0], detected_vehicles[j].bbox[1]}, cv::Point2f{detected_vehicles[j].bbox[2], detected_vehicles[j].bbox[3]});
                  if (b1.area() > b2.area())
                  {
                    if (config.logs_level <= userver::logging::Level::kTrace)
                      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                        "vstream_key = {}_{};  remove duplicate plate number from vehicle {}",
                        config.id_group, config.ext_id, i);
                    m = detected_vehicles[i].license_plates.erase(m);
                    --m;
                  } else
                  {
                    if (config.logs_level <= userver::logging::Level::kTrace)
                      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                        "vstream_key = {}_{};  remove duplicate plate number from vehicle {}",
                        config.id_group, config.ext_id, j);
                    n = detected_vehicles[j].license_plates.erase(n);
                    --n;
                  }
                } else
                {
                  // remove plate from a vehicle with most elements
                  if (detected_vehicles[i].license_plates.size() > detected_vehicles[j].license_plates.size())
                  {
                    if (config.logs_level <= userver::logging::Level::kTrace)
                      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                        "vstream_key = {}_{};  remove duplicate plate number from vehicle {}",
                        config.id_group, config.ext_id, i);
                    m = detected_vehicles[i].license_plates.erase(m);
                    --m;
                  } else
                  {
                    if (config.logs_level <= userver::logging::Level::kTrace)
                      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                        "vstream_key = {}_{};  remove duplicate plate number from vehicle {}",
                        config.id_group, config.ext_id, j);
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
          for (const auto wa = convertToAbsolute(config.work_area, img_width, img_height); const auto& v : wa)
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

  bool isNumber(const char c)
  {
    return c >= '0' && c <= '9';
  }

  bool isLetter(const char c)
  {
    return c >= 'A' && c <= 'Z';
  }

  bool isLetterRU(const char c)
  {
    return std::string_view("ABCEHKMOPTXY").find(c) != std::string_view::npos;
  }

  bool isLetterBY(const char c)
  {
    return std::string_view("ABCEHIKMOPTX").find(c) != std::string_view::npos;
  }

  // format: L NNN LL NN or L NNN LL NNN
  std::string checkRU(std::string number)
  {
    std::erase_if(number, [](const char c)
      { return !isNumber(c) && !isLetterRU(c); });

    if (number.size() < 8 || number.size() > 9)
      return {};

    for (size_t i = 0; i < number.size(); ++i)
      if (i == 0 || i == 4 || i == 5)
      {
        if (!isLetterRU(number[i]))
          return {};
      } else
      {
        if (!isNumber(number[i]))
          return {};
      }

    return number;
  }

  // format: NNNN LL N
  std::string checkBY(std::string number)
  {
    while (number.size() > 7)
    {
      if (!isNumber(number[0]))
        number.erase(0, 1);
      else
        return {};
    }

    if (number.size() != 7)
      return {};

    for (size_t i = 0; i < number.size(); ++i)
      if (i < 4 || i > 5)
      {
        if (!isNumber(number[i]))
          return {};
      } else
        if (!isLetterBY(number[i]))
          return {};

    return number;
  }

  // format: NN LL NNN or NNN LL NN
  std::string checkAM(std::string number)
  {
    while (number.size() > 7)
    {
      if (!isNumber(number[0]))
        number.erase(0, 1);
      else
        return {};
    }

    if (number.size() != 7)
      return {};

    for (size_t i = 0; i < number.size(); ++i)
      if (i == 0 || i == 1 || i == 5 || i == 6)
      {
        if (!isNumber(number[i]))
          return {};
      } else if (i == 3)
        if (!isLetter(number[i]))
          return {};

    return number;
  }

  bool Workflow::isValidPlateNumber(PlateNumberData& plate_number)
  {
    if (plate_number.number.empty())
      return false;

    std::string res = checkRU(plate_number.number);
    if (!res.empty())
    {
      plate_number.number = std::move(res);
      if (plate_number.plate_class != PLATE_CLASS_RU_1 && plate_number.plate_class != PLATE_CLASS_RU_1A)
        plate_number.plate_class = PLATE_CLASS_RU_1;
      return true;
    }

    res = checkBY(plate_number.number);
    if (!res.empty())
    {
      plate_number.number = std::move(res);
      plate_number.plate_class = PLATE_CLASS_BY;
      return true;
    }

    res = checkAM(plate_number.number);
    if (!res.empty())
    {
      plate_number.number = std::move(res);
      plate_number.plate_class = PLATE_CLASS_AM;
      return true;
    }

    return false;
  }

  bool Workflow::doInferenceLpcNet(const cv::Mat& img, const VStreamConfig& config, std::vector<LicensePlate*>& detected_plates) const
  {
    std::vector<userver::engine::TaskWithResult<triton::client::Error>> tasks;
    tasks.reserve(detected_plates.size());

    std::vector<std::unique_ptr<tc::InferenceServerHttpClient>> triton_clients;
    triton_clients.resize(detected_plates.size());

    std::vector<tc::InferResult*> results;
    results.resize(detected_plates.size());

    std::vector<std::vector<uint8_t>> inputs_data;
    inputs_data.reserve(detected_plates.size());

    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
    input_ptrs.reserve(detected_plates.size());

    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_ptrs;
    output_ptrs.reserve(detected_plates.size());

    std::vector<tc::InferOptions> options;
    options.reserve(detected_plates.size());

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before inference LpcNet",
        config.id_group, config.ext_id);

    for (size_t pindex = 0; pindex < detected_plates.size(); ++pindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[pindex], config.lpc_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create inference client: {}",
          err.Message());
        return false;
      }

      auto& [bbox, confidence, kpts, plate_class, plate_numbers] = *detected_plates[pindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  before preprocess image {} for LpcNet",
          config.id_group, config.ext_id, pindex);
      cv::Rect roi(cv::Point{static_cast<int>(bbox[0]), static_cast<int>(bbox[1])},
        cv::Point{static_cast<int>(bbox[2]), static_cast<int>(bbox[3])});
      auto blob = prepareBlobForLpcNet(img(roi), config.lpc_net_input_width, config.lpc_net_input_height);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  after preprocess image {} for LpcNet",
          config.id_group, config.ext_id, pindex);

      const auto* raw = blob.ptr<uint8_t>();
      const auto byte_size = blob.total() * blob.elemSize();
      inputs_data.emplace_back(raw, raw + byte_size);
      std::vector<int64_t> shape = {1, 3, config.lpc_net_input_height, config.lpc_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.lpc_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create input data: {}",
          err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.lpc_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create output data: {}",
          err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[pindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to set up input data: {}",
          err.Message());
        return false;
      }
      options.emplace_back(config.lpc_net_model_name);
      options.back().model_version_ = "";

      // inference timeout in microseconds
      options.back().client_timeout_ = std::chrono::duration_cast<std::chrono::microseconds>(config.inference_timeout).count();

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&triton_clients, &results, &options, &input_ptrs, &output_ptrs, pindex]
        {
          return triton_clients[pindex]->Infer(&results[pindex], options[pindex], {input_ptrs[pindex].get()}, {output_ptrs[pindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after inference LpcNet",
        config.id_group, config.ext_id);

    bool is_ok = false;
    for (size_t pindex = 0; pindex < detected_plates.size(); ++pindex)
    {
      if (auto err = tasks[pindex].Get(); !err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to send inference request (vindex = {}): {}",
          pindex, err.Message());
        continue;
      }

      const std::shared_ptr<tc::InferResult> result_ptr(results[pindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to receive inference result (vindex = {}): {}",
          pindex, result_ptr->RequestStatus().Message());
        continue;
      }

      const float* data;
      size_t data_size;
      result_ptr->RawData(config.vc_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      std::vector<float> scores;
      scores.assign(data, data + static_cast<int>(data_size / sizeof(float)));

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  pindex = {};  softmax scores: ",
          config.id_group, config.ext_id, pindex) << scores;

      scores = softMax(scores);

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  pindex = {};  bbox = [{:.2f}, {:.2f}, {:.2f}, {:.2f}];  scores: ",
          config.id_group, config.ext_id, pindex,
          detected_plates[pindex]->bbox[0], detected_plates[pindex]->bbox[1], detected_plates[pindex]->bbox[2], detected_plates[pindex]->bbox[3]) << scores;

      // determine the index of the maximum element
      int32_t max_index = 0;
      for (size_t i = 1; i < scores.size(); ++i)
      {
        if (scores[i] > scores[max_index])
          max_index = static_cast<int32_t>(i);
      }
      detected_plates[pindex]->plate_class_common = max_index;

      is_ok = true;
    }

    return is_ok;
  }

  bool Workflow::doInferenceLprNet(const cv::Mat& img, const VStreamConfig& config, std::vector<LicensePlate*>& detected_plates)
  {
    std::vector labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
      "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ""};

    // scale values by plate class to calculate height of an image after perspective transformation
    std::vector scale_height_by_plate_class = {
      112.0f / 520.f,   // 0 - Russia type 1
      170.0f / 290.0f,  // 1 - Russia type 1A
      112.0f / 520.f,   // 2 - Belarus
      112.0f / 520.f,   // 3 - Armenia
    };

    std::vector<userver::engine::TaskWithResult<triton::client::Error>> tasks;
    tasks.reserve(detected_plates.size());

    std::vector<std::unique_ptr<tc::InferenceServerHttpClient>> triton_clients;
    triton_clients.resize(detected_plates.size());

    std::vector<tc::InferResult*> results;
    results.resize(detected_plates.size());

    std::vector<std::vector<uint8_t>> inputs_data;
    inputs_data.reserve(detected_plates.size());

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
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  before inference LPRNet",
        config.id_group, config.ext_id);

    for (size_t pindex = 0; pindex < detected_plates.size(); ++pindex)
    {
      auto err = tc::InferenceServerHttpClient::Create(&triton_clients[pindex], config.lpr_net_inference_server, false);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create inference client: {}",
          err.Message());
        return false;
      }
      auto& [bbox, confidence, kpts, plate_class, plate_numbers] = *detected_plates[pindex];
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  before preprocess image {} for LPRNet",
          config.id_group, config.ext_id, pindex);

      const auto safe_plate_class = std::clamp(plate_class, 0, static_cast<int32_t>(scale_height_by_plate_class.size() - 1));
      auto src = cv::Mat(4, 2, CV_32F, kpts);
      std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(config.lpr_net_input_width - 1), 0.0f},
        {static_cast<float>(config.lpr_net_input_width - 1), static_cast<float>(config.lpr_net_input_width) * scale_height_by_plate_class[safe_plate_class] - 1},
        {0.0f, static_cast<float>(config.lpr_net_input_width) * scale_height_by_plate_class[safe_plate_class] - 1}};
      auto transform_mat = cv::getPerspectiveTransform(src, dst);
      cv::Mat lp_image;
      cv::warpPerspective(img, lp_image, transform_mat, {config.lpr_net_input_width, static_cast<int>(static_cast<float>(config.lpr_net_input_width) * scale_height_by_plate_class[safe_plate_class])});

      // for test
      // cv::imwrite(absl::Substitute("pp_$1_$0.png", pindex, config.ext_id), lp_image);

      auto blob = prepareBlobForYOLO(lp_image, config.lpr_net_input_width, config.lpr_net_input_height, shifts[pindex],
        scales[pindex]);

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  after preprocess image {} for LPRNet",
          config.id_group, config.ext_id, pindex);

      const auto* raw = blob.ptr<uint8_t>();
      const auto byte_size = blob.total() * blob.elemSize();
      inputs_data.emplace_back(raw, raw + byte_size);
      std::vector<int64_t> shape = {1, 3, config.lpr_net_input_height, config.lpr_net_input_width};
      tc::InferInput* input;
      err = tc::InferInput::Create(&input, config.lpr_net_input_tensor_name, shape, "FP32");
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create input data: {}",
          err.Message());
        return false;
      }
      input_ptrs.emplace_back(input);

      tc::InferRequestedOutput* output;
      err = tc::InferRequestedOutput::Create(&output, config.lpr_net_output_tensor_name);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to create output data: {}",
          err.Message());
        return false;
      }
      output_ptrs.emplace_back(output);

      err = input_ptrs.back()->AppendRaw(inputs_data[pindex]);
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to set up input data: {}",
          err.Message());
        return false;
      }

      options.emplace_back(config.lpr_net_model_name);
      options.back().model_version_ = "";

      // inference timeout in microseconds
      options.back().client_timeout_ = std::chrono::duration_cast<std::chrono::microseconds>(config.inference_timeout).count();

      tasks.emplace_back(AsyncNoSpan(fs_task_processor_,
        [&triton_clients, &results, &options, &input_ptrs, &output_ptrs, pindex]
        {
          return triton_clients[pindex]->Infer(&results[pindex], options[pindex], {input_ptrs[pindex].get()}, {output_ptrs[pindex].get()});
        }));
    }
    WaitAllChecked(tasks);

    if (config.logs_level <= userver::logging::Level::kTrace)
      USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
        "vstream_key = {}_{};  after inference LPRNet",
        config.id_group, config.ext_id);

    bool is_ok = false;
    for (size_t pindex = 0; pindex < detected_plates.size(); ++pindex)
    {
      auto err = tasks[pindex].Get();
      if (!err.IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to send inference request (vindex = {}): {}",
          pindex, err.Message());
        continue;
      }

      std::shared_ptr<tc::InferResult> result_ptr(results[pindex]);
      if (!result_ptr->RequestStatus().IsOk())
      {
        LOG_ERROR_TO(logger_,
          "Error! Unable to receive inference result (vindex = {}): {}",
          pindex, err.Message());
        continue;
      }

      auto& plate = *detected_plates[pindex];
      const float* data;
      size_t data_size;
      result_ptr->RawData(config.lpr_net_output_tensor_name, reinterpret_cast<const uint8_t**>(&data), &data_size);

      std::vector<CharData> chars_data;
      auto num_rows = 300;
      auto num_cols = 6;
      auto bbox_index = 0;
      auto conf_index = bbox_index + 4;
      auto class_index = conf_index + 1;
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  char score threshold: {:.2f}",
          config.id_group, config.ext_id, config.char_score);
      for (auto i = 0; i < num_rows; ++i)
        if (const float conf = data[conf_index + num_cols * i]; conf > config.char_score)
        {
          chars_data.emplace_back();
          auto& [bbox, confidence, char_class, plate_class] = chars_data.back();
          const float left = data[bbox_index + 0 + num_cols * i];
          const float top = data[bbox_index + 1 + num_cols * i];
          const float right = data[bbox_index + 2 + num_cols * i];
          const float bottom = data[bbox_index + 3 + num_cols * i];
          const float inv_scale = 1.0f / static_cast<float>(scales[pindex]);
          bbox[0] = (left - shifts[pindex].x) * inv_scale;
          bbox[1] = (top - shifts[pindex].y) * inv_scale;
          bbox[2] = (right - shifts[pindex].x) * inv_scale;
          bbox[3] = (bottom - shifts[pindex].y) * inv_scale;

          confidence = conf;
          char_class = static_cast<int32_t>(data[class_index + num_cols * i]);
          plate_class = plate.plate_class_common;
        }

      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  before nms_chars count (pindex = {}): {}",
          config.id_group, config.ext_id, pindex, chars_data.size());
      nms_chars(chars_data, config.char_iou_threshold);
      if (config.logs_level <= userver::logging::Level::kTrace)
        USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
          "vstream_key = {}_{};  after nms_chars count (pindex = {}): {}",
          config.id_group, config.ext_id, pindex, chars_data.size());

      // assembling license plate numbers from char data
      std::ranges::sort(chars_data, cmp_chars_position);
      HashSet<size_t> used_indices;
      plate.plate_numbers.push_back({plate.plate_class_common, "", 1.0f});
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
        [this, &config](auto& item)
        {
          auto is_valid = isValidPlateNumber(item);
          if (!is_valid)
            if (config.logs_level <= userver::logging::Level::kTrace)
              USERVER_IMPL_LOG_TO(logger_, userver::logging::Level::kTrace,
                "vstream_key = {}_{};  invalid plate number: {}",
                config.id_group, config.ext_id, item.number);
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
