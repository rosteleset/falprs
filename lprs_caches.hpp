#pragma once

#include <string>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <opencv2/opencv.hpp>
#include <userver/cache/base_postgres_cache.hpp>

namespace Lprs
{
  template <typename Key, typename Value>
  using HashMap = absl::flat_hash_map<Key, Value>;

  template <typename T>
  using HashSet = absl::flat_hash_set<T>;

  namespace ConfigParams
  {
    // Local
    inline static constexpr auto SECTION_NAME = "config";
    inline static constexpr auto ALLOW_GROUP_ID_WITHOUT_AUTH = "allow-group-id-without-auth";
    inline static constexpr auto BAN_MAINTENANCE_INTERVAL = "ban-maintenance-interval";
    inline static constexpr auto EVENTS_LOG_MAINTENANCE_INTERVAL = "events-log-maintenance-interval";
    inline static constexpr auto EVENTS_LOG_TTL = "events-log-ttl";
    inline static constexpr auto EVENTS_SCREENSHOTS_PATH = "screenshots-path";
    inline static constexpr auto EVENTS_SCREENSHOTS_URL_PREFIX = "screenshots-url-prefix";
    inline static constexpr auto FAILED_PATH = "failed-path";
    inline static constexpr auto FAILED_TTL = "failed-ttl";

    // Video stream
    inline static constexpr auto CALLBACK_TIMEOUT = "callback-timeout";
    inline static constexpr auto VD_NET_INFERENCE_SERVER = "vd-net-inference-server";
    inline static constexpr auto VD_NET_MODEL_NAME = "vd-net-model-name";
    inline static constexpr auto VD_NET_INPUT_WIDTH = "vd-net-input-width";
    inline static constexpr auto VD_NET_INPUT_HEIGHT = "vd-net-input-height";
    inline static constexpr auto VD_NET_INPUT_TENSOR_NAME = "vd-net-input-tensor-name";
    inline static constexpr auto VD_NET_OUTPUT_TENSOR_NAME = "vd-net-output-tensor-name";
    inline static constexpr auto VC_NET_INFERENCE_SERVER = "vd-net-inference-server";
    inline static constexpr auto VC_NET_MODEL_NAME = "vc-net-model-name";
    inline static constexpr auto VC_NET_INPUT_WIDTH = "vc-net-input-width";
    inline static constexpr auto VC_NET_INPUT_HEIGHT = "vc-net-input-height";
    inline static constexpr auto VC_NET_INPUT_TENSOR_NAME = "vc-net-input-tensor-name";
    inline static constexpr auto VC_NET_OUTPUT_TENSOR_NAME = "vc-net-output-tensor-name";
    inline static constexpr auto LPD_NET_INFERENCE_SERVER = "lpd-net-inference-server";
    inline static constexpr auto LPD_NET_MODEL_NAME = "lpd-net-model-name";
    inline static constexpr auto LPD_NET_INPUT_WIDTH = "lpd-net-input-width";
    inline static constexpr auto LPD_NET_INPUT_HEIGHT = "lpd-net-input-height";
    inline static constexpr auto LPD_NET_INPUT_TENSOR_NAME = "lpd-net-input-tensor-name";
    inline static constexpr auto LPD_NET_OUTPUT_TENSOR_NAME = "lpd-net-output-tensor-name";
    inline static constexpr auto LPR_NET_INFERENCE_SERVER = "lpr-net-inference-server";
    inline static constexpr auto LPR_NET_MODEL_NAME = "lpr-net-model-name";
    inline static constexpr auto LPR_NET_INPUT_WIDTH = "lpr-net-input-width";
    inline static constexpr auto LPR_NET_INPUT_HEIGHT = "lpr-net-input-height";
    inline static constexpr auto LPR_NET_INPUT_TENSOR_NAME = "lpr-net-input-tensor-name";
    inline static constexpr auto LPR_NET_OUTPUT_TENSOR_NAME = "lpr-net-output-tensor-name";
    inline static constexpr auto VEHICLE_CONFIDENCE = "vehicle-confidence";
    inline static constexpr auto VEHICLE_IOU_THRESHOLD = "vehicle-iou-threshold";
    inline static constexpr auto VEHICLE_AREA_RATIO_THRESHOLD = "vehicle-area-ratio-threshold";
    inline static constexpr auto SPECIAL_CONFIDENCE = "special-confidence";
    inline static constexpr auto PLATE_CONFIDENCE = "plate-confidence";
    inline static constexpr auto CHAR_SCORE = "char-score";
    inline static constexpr auto CHAR_IOU_THRESHOLD = "char-iou-threshold";
    inline static constexpr auto MAX_CAPTURE_ERROR_COUNT = "max-capture-error-count";
    inline static constexpr auto CAPTURE_TIMEOUT = "capture-timeout";
    inline static constexpr auto EVENT_LOG_BEFORE = "event-log-before";
    inline static constexpr auto EVENT_LOG_AFTER = "event-log-after";
    inline static constexpr auto DELAY_BETWEEN_FRAMES = "delay-between-frames";
    inline static constexpr auto BAN_DURATION = "ban-duration";
    inline static constexpr auto BAN_DURATION_AREA = "ban-duration-area";
    inline static constexpr auto BAN_IOU_THRESHOLD = "ban-iou-threshold";
    inline static constexpr auto DELAY_AFTER_ERROR = "delay-after-error";
    inline static constexpr auto LOGS_LEVEL = "logs-level";
    inline static constexpr auto MIN_PLATE_HEIGHT = "min-plate-height";
    inline static constexpr auto FLAG_SAVE_FAILED = "flag-save-failed";
    inline static constexpr auto FLAG_PROCESS_SPECIAL = "flag-process-special";

    // Video stream specific params
    inline static constexpr auto SCREENSHOT_URL = "screenshot-url";
    inline static constexpr auto CALLBACK_URL = "callback-url";
    inline static constexpr auto WORK_AREA = "work-area";
  }  // namespace ConfigParams

  struct VStreamConfig
  {
    std::string vd_net_inference_server{"127.0.0.1:8000"};
    std::string vd_net_model_name{"vdnet_yolo"};
    int32_t vd_net_input_width = 640;
    int32_t vd_net_input_height = 640;
    std::string vd_net_input_tensor_name{"images"};
    std::string vd_net_output_tensor_name{"output0"};

    std::string vc_net_inference_server{"127.0.0.1:8000"};
    std::string vc_net_model_name{"vc_genet"};
    int32_t vc_net_input_width = 224;
    int32_t vc_net_input_height = 224;
    std::string vc_net_input_tensor_name{"input"};
    std::string vc_net_output_tensor_name{"output"};

    std::string lpd_net_inference_server{"127.0.0.1:8000"};
    std::string lpd_net_model_name{"lpdnet_yolo"};
    int32_t lpd_net_input_width = 640;
    int32_t lpd_net_input_height = 640;
    std::string lpd_net_input_tensor_name{"images"};
    std::string lpd_net_output_tensor_name{"output0"};

    std::string lpr_net_inference_server{"127.0.0.1:8000"};
    std::string lpr_net_model_name{"lprnet_yolo"};
    int32_t lpr_net_input_width = 160;
    int32_t lpr_net_input_height = 160;
    std::string lpr_net_input_tensor_name{"images"};
    std::string lpr_net_output_tensor_name{"output0"};

    std::chrono::milliseconds callback_timeout{std::chrono::seconds{2}};
    float plate_confidence{0.6};
    float char_score{0.4};
    float char_iou_threshold{0.7};
    short max_capture_error_count{3};
    float vehicle_confidence{0.6};
    float vehicle_iou_threshold{0.45};
    float vehicle_area_ratio_threshold{0.01};
    float special_confidence{0.7};
    std::chrono::milliseconds capture_timeout{std::chrono::seconds{2}};
    std::chrono::milliseconds event_log_before{std::chrono::seconds{10}};
    std::chrono::milliseconds event_log_after{std::chrono::seconds{5}};
    std::chrono::milliseconds delay_between_frames{std::chrono::seconds{1}};
    std::chrono::milliseconds ban_duration{std::chrono::seconds{30}};
    std::chrono::milliseconds ban_duration_area{std::chrono::hours{12}};
    float ban_iou_threshold{0.9};
    std::chrono::milliseconds delay_after_error{std::chrono::seconds{30}};
    userver::logging::Level logs_level{userver::logging::Level::kInfo};
    int32_t min_plate_height{0};
    std::vector<std::vector<cv::Point2f>> work_area;
    bool flag_save_failed{false};
    bool flag_process_special{false};

    // additional data
    int32_t id_group{};
    int32_t id_vstream{};
    std::string ext_id;
    std::string screenshot_url;
    std::string callback_url;
  };

  static VStreamConfig updateVStreamConfig(const userver::formats::json::Value& json)
  {
    VStreamConfig config;

    config.vd_net_inference_server = json[ConfigParams::VD_NET_INFERENCE_SERVER].As<decltype(config.vd_net_inference_server)>(config.vd_net_inference_server);
    config.vd_net_model_name = json[ConfigParams::VD_NET_MODEL_NAME].As<decltype(config.vd_net_model_name)>(config.vd_net_model_name);
    config.vd_net_input_width = json[ConfigParams::VD_NET_INPUT_WIDTH].As<decltype(config.vd_net_input_width)>(config.vd_net_input_width);
    config.vd_net_input_height = json[ConfigParams::VD_NET_INPUT_HEIGHT].As<decltype(config.vd_net_input_height)>(config.vd_net_input_height);
    config.vd_net_input_tensor_name = json[ConfigParams::VD_NET_INPUT_TENSOR_NAME].As<decltype(config.vd_net_input_tensor_name)>(config.vd_net_input_tensor_name);
    config.vd_net_output_tensor_name = json[ConfigParams::VD_NET_OUTPUT_TENSOR_NAME].As<decltype(config.vd_net_output_tensor_name)>(config.vd_net_output_tensor_name);

    config.vc_net_inference_server = json[ConfigParams::VC_NET_INFERENCE_SERVER].As<decltype(config.vc_net_inference_server)>(config.vc_net_inference_server);
    config.vc_net_model_name = json[ConfigParams::VC_NET_MODEL_NAME].As<decltype(config.vc_net_model_name)>(config.vc_net_model_name);
    config.vc_net_input_width = json[ConfigParams::VC_NET_INPUT_WIDTH].As<decltype(config.vc_net_input_width)>(config.vc_net_input_width);
    config.vc_net_input_height = json[ConfigParams::VC_NET_INPUT_HEIGHT].As<decltype(config.vc_net_input_height)>(config.vc_net_input_height);
    config.vc_net_input_tensor_name = json[ConfigParams::VC_NET_INPUT_TENSOR_NAME].As<decltype(config.vc_net_input_tensor_name)>(config.vc_net_input_tensor_name);
    config.vc_net_output_tensor_name = json[ConfigParams::VC_NET_OUTPUT_TENSOR_NAME].As<decltype(config.vc_net_output_tensor_name)>(config.vc_net_output_tensor_name);

    config.lpd_net_inference_server = json[ConfigParams::LPD_NET_INFERENCE_SERVER].As<decltype(config.lpd_net_inference_server)>(config.lpd_net_inference_server);
    config.lpd_net_model_name = json[ConfigParams::LPD_NET_MODEL_NAME].As<decltype(config.lpd_net_model_name)>(config.lpd_net_model_name);
    config.lpd_net_input_width = json[ConfigParams::LPD_NET_INPUT_WIDTH].As<decltype(config.lpd_net_input_width)>(config.lpd_net_input_width);
    config.lpd_net_input_height = json[ConfigParams::LPD_NET_INPUT_HEIGHT].As<decltype(config.lpd_net_input_height)>(config.lpd_net_input_height);
    config.lpd_net_input_tensor_name = json[ConfigParams::LPD_NET_INPUT_TENSOR_NAME].As<decltype(config.lpd_net_input_tensor_name)>(config.lpd_net_input_tensor_name);
    config.lpd_net_output_tensor_name = json[ConfigParams::LPD_NET_OUTPUT_TENSOR_NAME].As<decltype(config.lpd_net_output_tensor_name)>(config.lpd_net_output_tensor_name);

    config.lpr_net_inference_server = json[ConfigParams::LPR_NET_INFERENCE_SERVER].As<decltype(config.lpr_net_inference_server)>(config.lpr_net_inference_server);
    config.lpr_net_model_name = json[ConfigParams::LPR_NET_MODEL_NAME].As<decltype(config.lpr_net_model_name)>(config.lpr_net_model_name);
    config.lpr_net_input_width = json[ConfigParams::LPR_NET_INPUT_WIDTH].As<decltype(config.lpr_net_input_width)>(config.lpr_net_input_width);
    config.lpr_net_input_height = json[ConfigParams::LPR_NET_INPUT_HEIGHT].As<decltype(config.lpr_net_input_height)>(config.lpr_net_input_height);
    config.lpr_net_input_tensor_name = json[ConfigParams::LPR_NET_INPUT_TENSOR_NAME].As<decltype(config.lpr_net_input_tensor_name)>(config.lpr_net_input_tensor_name);
    config.lpr_net_output_tensor_name = json[ConfigParams::LPR_NET_OUTPUT_TENSOR_NAME].As<decltype(config.lpr_net_output_tensor_name)>(config.lpr_net_output_tensor_name);

    config.callback_timeout = json.HasMember(ConfigParams::CALLBACK_TIMEOUT)
                                ? userver::utils::StringToDuration(json[ConfigParams::CALLBACK_TIMEOUT].As<std::string>())
                                : config.callback_timeout;
    config.plate_confidence = json[ConfigParams::PLATE_CONFIDENCE].As<decltype(config.plate_confidence)>(config.plate_confidence);
    config.char_score = json[ConfigParams::CHAR_SCORE].As<decltype(config.char_score)>(config.char_score);
    config.char_iou_threshold = json[ConfigParams::CHAR_IOU_THRESHOLD].As<decltype(config.char_iou_threshold)>(config.char_iou_threshold);
    config.max_capture_error_count = json[ConfigParams::MAX_CAPTURE_ERROR_COUNT].As<decltype(config.max_capture_error_count)>(config.max_capture_error_count);
    config.vehicle_confidence = json[ConfigParams::VEHICLE_CONFIDENCE].As<decltype(config.vehicle_confidence)>(config.vehicle_confidence);
    config.vehicle_iou_threshold = json[ConfigParams::VEHICLE_IOU_THRESHOLD].As<decltype(config.vehicle_iou_threshold)>(config.vehicle_iou_threshold);
    config.vehicle_area_ratio_threshold = json[ConfigParams::VEHICLE_AREA_RATIO_THRESHOLD].As<decltype(config.vehicle_area_ratio_threshold)>(config.vehicle_area_ratio_threshold);
    config.special_confidence = json[ConfigParams::SPECIAL_CONFIDENCE].As<decltype(config.special_confidence)>(config.special_confidence);
    config.capture_timeout = json.HasMember(ConfigParams::CAPTURE_TIMEOUT)
                               ? userver::utils::StringToDuration(json[ConfigParams::CAPTURE_TIMEOUT].As<std::string>())
                               : config.capture_timeout;
    config.event_log_before = json.HasMember(ConfigParams::EVENT_LOG_BEFORE)
                                ? userver::utils::StringToDuration(json[ConfigParams::EVENT_LOG_BEFORE].As<std::string>())
                                : config.event_log_before;
    config.event_log_after = json.HasMember(ConfigParams::EVENT_LOG_AFTER)
                               ? userver::utils::StringToDuration(json[ConfigParams::EVENT_LOG_AFTER].As<std::string>())
                               : config.event_log_after;
    config.delay_between_frames = json.HasMember(ConfigParams::DELAY_BETWEEN_FRAMES)
                                    ? userver::utils::StringToDuration(json[ConfigParams::DELAY_BETWEEN_FRAMES].As<std::string>())
                                    : config.delay_between_frames;
    config.ban_duration = json.HasMember(ConfigParams::BAN_DURATION)
                            ? userver::utils::StringToDuration(json[ConfigParams::BAN_DURATION].As<std::string>())
                            : config.ban_duration;
    config.ban_duration_area = json.HasMember(ConfigParams::BAN_DURATION_AREA)
                                 ? userver::utils::StringToDuration(json[ConfigParams::BAN_DURATION_AREA].As<std::string>())
                                 : config.ban_duration_area;
    config.ban_iou_threshold = json[ConfigParams::BAN_IOU_THRESHOLD].As<decltype(config.ban_iou_threshold)>(config.ban_iou_threshold);
    config.delay_after_error = json.HasMember(ConfigParams::DELAY_AFTER_ERROR)
                                 ? userver::utils::StringToDuration(json[ConfigParams::DELAY_AFTER_ERROR].As<std::string>())
                                 : config.delay_after_error;
    config.min_plate_height = json[ConfigParams::MIN_PLATE_HEIGHT].As<decltype(config.min_plate_height)>(config.min_plate_height);
    config.flag_save_failed = json[ConfigParams::FLAG_SAVE_FAILED].As<decltype(config.flag_save_failed)>(config.flag_save_failed);
    config.flag_process_special = json[ConfigParams::FLAG_PROCESS_SPECIAL].As<decltype(config.flag_process_special)>(config.flag_process_special);
    config.screenshot_url = json[ConfigParams::SCREENSHOT_URL].As<decltype(config.screenshot_url)>(config.screenshot_url);
    config.callback_url = json[ConfigParams::CALLBACK_URL].As<decltype(config.callback_url)>(config.callback_url);
    config.logs_level = json.HasMember(ConfigParams::LOGS_LEVEL)
                          ? userver::logging::LevelFromString(json[ConfigParams::LOGS_LEVEL].As<std::string>())
                          : config.logs_level;
    if (json.HasMember(ConfigParams::WORK_AREA) && json[ConfigParams::WORK_AREA].IsArray())
      try
      {
        for (auto v = json[ConfigParams::WORK_AREA].As<std::vector<std::vector<std::vector<float>>>>(); auto& i : v)
        {
          std::vector<cv::Point2f> polygon;
          polygon.reserve(i.size());
          for (auto& j : i)
            polygon.emplace_back(j[0], j[1]);
          config.work_area.push_back(std::move(polygon));
        }
      } catch (...)
      {
        config.work_area = {};
      }

    return config;
  }

  // Video stream groups' cache
  struct VStreamGroups
  {
    std::string auth_token;
    int32_t id_group;
  };

  struct GroupsCachePolicy
  {
    static constexpr std::string_view kName = "lprs-groups-pg-cache";

    using ValueType = VStreamGroups;
    static constexpr auto kKeyMember = &VStreamGroups::auth_token;
    static constexpr auto kQuery = "select auth_token::varchar, id_group from vstream_groups";
    static constexpr const char* kUpdatedField = nullptr;
    using CacheContainer = HashMap<std::string, VStreamGroups>;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using GroupsCache = userver::components::PostgreCache<GroupsCachePolicy>;

  // Video streams configuration cache
  struct VStreamConfigData
  {
    std::string unique_key;
    int32_t id_group;
    int32_t id_vstream;
    std::string ext_id;
    std::optional<userver::formats::json::Value> config;
  };

  class VStreamConfigContainer
  {
  public:
    void insert_or_assign(const std::string& key, VStreamConfigData&& item)
    {
      VStreamConfig config{};
      if (item.config)
        config = updateVStreamConfig(*item.config);
      config.id_group = item.id_group;
      config.id_vstream = item.id_vstream;
      config.ext_id = item.ext_id;
      data_[key] = std::move(config);
    }

    static size_t size()
    {
      return 0;
    }

    [[nodiscard]] const auto& getData() const
    {
      return data_;
    }

  private:
    HashMap<std::string, VStreamConfig> data_;
  };

  struct VStreamsConfigCachePolicy
  {
    static constexpr std::string_view kName = "lprs-vstreams-config-pg-cache";

    using ValueType = VStreamConfigData;
    static constexpr auto kKeyMember = &VStreamConfigData::unique_key;
    static constexpr auto kQuery = R"__SQL__(
      select
        concat(vs.id_group, '_', vs.ext_id) unique_key,
        vs.id_group,
        vs.id_vstream,
        vs.ext_id,
        coalesce(d.config, '{}') || coalesce(vs.config, '{}') config
      from
        vstreams vs
        left join default_vstream_config d
          on d.id_group = vs.id_group
    )__SQL__";
    static constexpr const char* kUpdatedField = nullptr;
    using CacheContainer = VStreamConfigContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using VStreamsConfigCache = userver::components::PostgreCache<VStreamsConfigCachePolicy>;

  // Video streams and their groups cache
  struct VStreamGroup
  {
    int32_t id_vstream;
    int32_t id_group;
  };

  struct VStreamGroupCachePolicy
  {
    static constexpr std::string_view kName = "lprs-vstream-group-pg-cache";

    using ValueType = VStreamGroup;
    static constexpr auto kKeyMember = &VStreamGroup::id_vstream;
    static constexpr auto kQuery = "select id_vstream, id_group from vstreams";
    static constexpr const char* kUpdatedField = nullptr;
    using CacheContainer = HashMap<int32_t, VStreamGroup>;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using VStreamGroupCache = userver::components::PostgreCache<VStreamGroupCachePolicy>;
}  // namespace Lprs
