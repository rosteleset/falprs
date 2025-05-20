#pragma once

#include <string>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <opencv2/opencv.hpp>
#include <userver/cache/base_postgres_cache.hpp>
#include <userver/storages/postgres/io/bytea.hpp>

#include "converters.hpp"

namespace Frs
{
  template <typename Key, typename Value>
  using HashMap = absl::flat_hash_map<Key, Value>;

  template <typename T>
  using HashSet = absl::flat_hash_set<T>;

  typedef cv::Mat FaceDescriptor;

  namespace ConfigParams
  {
    // Local
    inline static constexpr auto SECTION_NAME = "config";
    inline static constexpr auto ALLOW_GROUP_ID_WITHOUT_AUTH = "allow-group-id-without-auth";
    inline static constexpr auto SCREENSHOTS_PATH = "screenshots-path";
    inline static constexpr auto SCREENSHOTS_URL_PREFIX = "screenshots-url-prefix";
    inline static constexpr auto EVENTS_PATH = "events-path";
    inline static constexpr auto CLEAR_OLD_LOG_FACES = "clear-old-log-faces";
    inline static constexpr auto FLAG_DELETED_MAINTENANCE_INTERVAL = "flag-deleted-maintenance-interval";
    inline static constexpr auto COPY_EVENTS_MAINTENANCE_INTERVAL = "copy-events-maintenance-interval";
    inline static constexpr auto CLEAR_OLD_EVENTS = "clear-old-events";
    inline static constexpr auto LOG_FACES_TTL = "log-faces-ttl";
    inline static constexpr auto FLAG_DELETED_TTL = "flag-deleted-ttl";
    inline static constexpr auto EVENTS_TTL = "events-ttl";

    // Common
    inline static constexpr auto CALLBACK_TIMEOUT = "callback-timeout";
    inline static constexpr auto FLAG_COPY_EVENT_DATA = "flag-copy-event-data";
    inline static constexpr auto DNN_FD_MODEL_NAME = "dnn-fd-model-name";
    inline static constexpr auto DNN_FD_INPUT_WIDTH = "dnn-fd-input-width";
    inline static constexpr auto DNN_FD_INPUT_HEIGHT = "dnn-fd-input-height";
    inline static constexpr auto DNN_FD_INPUT_TENSOR_NAME = "dnn-fd-input-tensor-name";
    inline static constexpr auto DNN_FC_MODEL_NAME = "dnn-fc-model-name";
    inline static constexpr auto DNN_FC_INPUT_WIDTH = "dnn-fc-input-width";
    inline static constexpr auto DNN_FC_INPUT_HEIGHT = "dnn-fc-input-height";
    inline static constexpr auto DNN_FC_INPUT_TENSOR_NAME = "dnn-fc-input-tensor-name";
    inline static constexpr auto DNN_FC_OUTPUT_TENSOR_NAME = "dnn-fc-output-tensor-name";
    inline static constexpr auto DNN_FC_OUTPUT_SIZE = "dnn-fc-output-size";
    inline static constexpr auto DNN_FR_MODEL_NAME = "dnn-fr-model-name";
    inline static constexpr auto DNN_FR_INPUT_WIDTH = "dnn-fr-input-width";
    inline static constexpr auto DNN_FR_INPUT_HEIGHT = "dnn-fr-input-height";
    inline static constexpr auto DNN_FR_INPUT_TENSOR_NAME = "dnn-fr-input-tensor-name";
    inline static constexpr auto DNN_FR_OUTPUT_TENSOR_NAME = "dnn-fr-output-tensor-name";
    inline static constexpr auto DNN_FR_OUTPUT_SIZE = "dnn-fr-output-size";
    inline static constexpr auto COMMENTS_BLURRY_FACE = "comments-blurry-face";
    inline static constexpr auto COMMENTS_DESCRIPTOR_CREATION_ERROR = "comments-descriptor-creation-error";
    inline static constexpr auto COMMENTS_DESCRIPTOR_EXISTS = "comments-descriptor-exists";
    inline static constexpr auto COMMENTS_INFERENCE_ERROR = "comments-inference-error";
    inline static constexpr auto COMMENTS_NEW_DESCRIPTOR = "comments-new-descriptor";
    inline static constexpr auto COMMENTS_NO_FACES = "comments-no-faces";
    inline static constexpr auto COMMENTS_NON_FRONTAL_FACE = "comments-non-frontal-face";
    inline static constexpr auto COMMENTS_NON_NORMAL_FACE_CLASS = "comments-non-normal-face-class";
    inline static constexpr auto COMMENTS_PARTIAL_FACE = "comments-partial-face";
    inline static constexpr auto COMMENTS_URL_IMAGE_ERROR = "comments-url-image-error";
    inline static constexpr auto SG_MAX_DESCRIPTOR_COUNT = "sg-max-descriptor-count";

    // Video stream
    inline static constexpr auto BEST_QUALITY_INTERVAL_AFTER = "best-quality-interval-after";
    inline static constexpr auto BEST_QUALITY_INTERVAL_BEFORE = "best-quality-interval-before";
    inline static constexpr auto BLUR = "blur";
    inline static constexpr auto BLUR_MAX = "blur-max";
    inline static constexpr auto CAPTURE_TIMEOUT = "capture-timeout";
    inline static constexpr auto DELAY_AFTER_ERROR = "delay-after-error";
    inline static constexpr auto DELAY_BETWEEN_FRAMES = "delay-between-frames";
    inline static constexpr auto DNN_FD_INFERENCE_SERVER = "dnn-fd-inference-server";
    inline static constexpr auto DNN_FC_INFERENCE_SERVER = "dnn-fc-inference-server";
    inline static constexpr auto DNN_FR_INFERENCE_SERVER = "dnn-fr-inference-server";
    inline static constexpr auto FACE_CLASS_CONFIDENCE_THRESHOLD = "face-class-confidence";
    inline static constexpr auto FACE_CONFIDENCE_THRESHOLD = "face-confidence";
    inline static constexpr auto FACE_ENLARGE_SCALE = "face-enlarge-scale";
    inline static constexpr auto LOGS_LEVEL = "logs-level";
    inline static constexpr auto MARGIN = "margin";
    inline static constexpr auto MAX_CAPTURE_ERROR_COUNT = "max-capture-error-count";
    inline static constexpr auto OPEN_DOOR_DURATION = "open-door-duration";
    inline static constexpr auto TOLERANCE = "tolerance";
    inline static constexpr auto TITLE_HEIGHT_RATIO = "title-height-ratio";
    inline static constexpr auto CONF_OSD_DT_FORMAT = "osd-datetime-format";

    // Video stream specific params
    inline static constexpr auto TITLE = "title";
    inline static constexpr auto WORK_AREA = "work-area";
  }  // namespace ConfigParams

  struct CommonConfig
  {
    std::chrono::milliseconds callback_timeout{std::chrono::seconds{2}};
    bool flag_copy_event_data{false};
    std::string dnn_fd_model_name{"scrfd"};
    int32_t dnn_fd_input_width{320};
    int32_t dnn_fd_input_height{320};
    std::string dnn_fd_input_tensor_name{"input.1"};
    std::string dnn_fc_model_name{"genet"};
    int32_t dnn_fc_input_width{192};
    int32_t dnn_fc_input_height{192};
    std::string dnn_fc_input_tensor_name{"input.1"};
    std::string dnn_fc_output_tensor_name{"419"};
    int32_t dnn_fc_output_size{3};
    std::string dnn_fr_model_name{"arcface"};
    int32_t dnn_fr_input_width{112};
    int32_t dnn_fr_input_height{112};
    std::string dnn_fr_input_tensor_name{"input.1"};
    std::string dnn_fr_output_tensor_name{"683"};
    int32_t dnn_fr_output_size{512};
    std::string comments_blurry_face{"The face image is not clear enough for registration."};
    std::string comments_descriptor_creation_error{"Failed to register descriptor."};
    std::string comments_descriptor_exists{"The descriptor already exists."};
    std::string comments_inference_error{"Error: Triton Inference Server request failed."};
    std::string comments_new_descriptor{"A new descriptor has been created."};
    std::string comments_no_faces{"There are no faces in the image."};
    std::string comments_non_frontal_face{"The face in the image must be frontal."};
    std::string comments_non_normal_face_class{"A person wearing a mask or dark glasses."};
    std::string comments_partial_face{"The face must be fully visible in the image."};
    std::string comments_url_image_error{"Failed to receive image."};
    int32_t sg_max_descriptor_count{1000};
  };

  struct VStreamConfig
  {
    std::chrono::milliseconds best_quality_interval_after{std::chrono::seconds{2}};
    std::chrono::milliseconds best_quality_interval_before{std::chrono::seconds{5}};
    float blur{300.0};
    float blur_max{13'000};
    std::chrono::milliseconds capture_timeout{std::chrono::seconds{2}};
    std::chrono::milliseconds delay_after_error{std::chrono::seconds{30}};
    std::chrono::milliseconds delay_between_frames{std::chrono::seconds{1}};
    std::string dnn_fd_inference_server{"127.0.0.1:8000"};
    std::string dnn_fc_inference_server{"127.0.0.1:8000"};
    std::string dnn_fr_inference_server{"127.0.0.1:8000"};
    float face_class_confidence{0.7};
    float face_confidence{0.7};
    float face_enlarge_scale{1.5};
    userver::logging::Level logs_level{userver::logging::Level::kInfo};
    float margin{5.0};
    short max_capture_error_count{3};
    std::chrono::milliseconds open_door_duration{std::chrono::seconds{10}};
    float tolerance{0.5};
    std::string title;
    float title_height_ratio{0.033};
    std::string osd_dt_format{"%Y-%m-%d %H:%M:%S"};
    std::vector<float> work_area;

    // additional data
    int32_t id_group{};
    int32_t id_vstream{};
    std::string vstream_ext;
    std::string url;
    std::string callback_url;
  };

  static VStreamConfig updateVStreamConfig(const userver::formats::json::Value& json)
  {
    VStreamConfig config{};
    config.best_quality_interval_after = convertToDuration(json[ConfigParams::BEST_QUALITY_INTERVAL_AFTER], config.best_quality_interval_after);
    config.best_quality_interval_before = convertToDuration(json[ConfigParams::BEST_QUALITY_INTERVAL_BEFORE], config.best_quality_interval_before);
    config.blur = convertToNumber(json[ConfigParams::BLUR], config.blur);
    config.blur_max = convertToNumber(json[ConfigParams::BLUR_MAX], config.blur_max);
    config.capture_timeout = convertToDuration(json[ConfigParams::CAPTURE_TIMEOUT], config.capture_timeout);
    config.delay_after_error = convertToDuration(json[ConfigParams::DELAY_AFTER_ERROR], config.delay_after_error);
    config.delay_between_frames = convertToDuration(json[ConfigParams::DELAY_BETWEEN_FRAMES], config.delay_between_frames);
    config.dnn_fd_inference_server = convertToString(json[ConfigParams::DNN_FD_INFERENCE_SERVER], config.dnn_fd_inference_server);
    config.dnn_fc_inference_server = convertToString(json[ConfigParams::DNN_FC_INFERENCE_SERVER], config.dnn_fc_inference_server);
    config.dnn_fr_inference_server = convertToString(json[ConfigParams::DNN_FR_INFERENCE_SERVER], config.dnn_fr_inference_server);
    config.face_class_confidence = convertToNumber(json[ConfigParams::FACE_CLASS_CONFIDENCE_THRESHOLD], config.face_class_confidence);
    config.face_confidence = convertToNumber(json[ConfigParams::FACE_CONFIDENCE_THRESHOLD], config.face_confidence);
    config.face_enlarge_scale = convertToNumber(json[ConfigParams::FACE_ENLARGE_SCALE], config.face_enlarge_scale);
    config.logs_level = convertToLevel(json[ConfigParams::LOGS_LEVEL], config.logs_level);
    config.margin = convertToNumber(json[ConfigParams::MARGIN], config.margin);
    config.max_capture_error_count = convertToNumber(json[ConfigParams::MAX_CAPTURE_ERROR_COUNT], config.max_capture_error_count);
    config.open_door_duration = convertToDuration(json[ConfigParams::OPEN_DOOR_DURATION], config.open_door_duration);
    config.tolerance = convertToNumber(json[ConfigParams::TOLERANCE], config.tolerance);
    config.title = convertToString(json[ConfigParams::TITLE], config.title);
    config.title_height_ratio = convertToNumber(json[ConfigParams::TITLE_HEIGHT_RATIO], config.title_height_ratio);
    config.osd_dt_format = convertToString(json[ConfigParams::CONF_OSD_DT_FORMAT], config.osd_dt_format);
    if (json.HasMember(ConfigParams::WORK_AREA) && json[ConfigParams::WORK_AREA].IsArray())
      config.work_area = json[ConfigParams::WORK_AREA].As<decltype(config.work_area)>();

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
    static constexpr std::string_view kName = "frs-groups-pg-cache";

    using ValueType = VStreamGroups;
    static constexpr auto kKeyMember = &VStreamGroups::auth_token;
    static constexpr auto kQuery = "select auth_token::varchar, id_group from vstream_groups";
    static constexpr const char* kUpdatedField = nullptr;
    using CacheContainer = HashMap<std::string, VStreamGroups>;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using GroupsCache = userver::components::PostgreCache<GroupsCachePolicy>;

  // Common and default video stream configuration cache
  struct ConfigData
  {
    int32_t id_group{1};
    std::optional<userver::formats::json::Value> config;
  };

  class ConfigContainer
  {
  public:
    void insert_or_assign(int32_t, ConfigData&& item)
    {
      if (item.config)
      {
        // common config
        common_config_[item.id_group].callback_timeout = item.config->HasMember(ConfigParams::CALLBACK_TIMEOUT)
                                                           ? userver::utils::StringToDuration((*item.config)[ConfigParams::CALLBACK_TIMEOUT].As<std::string>())
                                                           : common_config_[item.id_group].callback_timeout;
        common_config_[item.id_group].flag_copy_event_data = (*item.config)[ConfigParams::FLAG_COPY_EVENT_DATA].As<decltype(common_config_[item.id_group].flag_copy_event_data)>(common_config_[item.id_group].flag_copy_event_data);

        common_config_[item.id_group].dnn_fd_model_name = (*item.config)[ConfigParams::DNN_FD_MODEL_NAME].As<decltype(common_config_[item.id_group].dnn_fd_model_name)>(common_config_[item.id_group].dnn_fd_model_name);
        common_config_[item.id_group].dnn_fd_input_width = (*item.config)[ConfigParams::DNN_FD_INPUT_WIDTH].As<decltype(common_config_[item.id_group].dnn_fd_input_width)>(common_config_[item.id_group].dnn_fd_input_width);
        common_config_[item.id_group].dnn_fd_input_height = (*item.config)[ConfigParams::DNN_FD_INPUT_HEIGHT].As<decltype(common_config_[item.id_group].dnn_fd_input_height)>(common_config_[item.id_group].dnn_fd_input_height);
        common_config_[item.id_group].dnn_fd_input_tensor_name = (*item.config)[ConfigParams::DNN_FD_INPUT_TENSOR_NAME].As<decltype(common_config_[item.id_group].dnn_fd_input_tensor_name)>(common_config_[item.id_group].dnn_fd_input_tensor_name);

        common_config_[item.id_group].dnn_fc_model_name = (*item.config)[ConfigParams::DNN_FC_MODEL_NAME].As<decltype(common_config_[item.id_group].dnn_fc_model_name)>(common_config_[item.id_group].dnn_fc_model_name);
        common_config_[item.id_group].dnn_fc_input_width = (*item.config)[ConfigParams::DNN_FC_INPUT_WIDTH].As<decltype(common_config_[item.id_group].dnn_fc_input_width)>(common_config_[item.id_group].dnn_fc_input_width);
        common_config_[item.id_group].dnn_fc_input_height = (*item.config)[ConfigParams::DNN_FC_INPUT_HEIGHT].As<decltype(common_config_[item.id_group].dnn_fc_input_height)>(common_config_[item.id_group].dnn_fc_input_height);
        common_config_[item.id_group].dnn_fc_input_tensor_name = (*item.config)[ConfigParams::DNN_FC_INPUT_TENSOR_NAME].As<decltype(common_config_[item.id_group].dnn_fc_input_tensor_name)>(common_config_[item.id_group].dnn_fc_input_tensor_name);
        common_config_[item.id_group].dnn_fc_output_tensor_name = (*item.config)[ConfigParams::DNN_FC_OUTPUT_TENSOR_NAME].As<decltype(common_config_[item.id_group].dnn_fc_output_tensor_name)>(common_config_[item.id_group].dnn_fc_output_tensor_name);
        common_config_[item.id_group].dnn_fc_output_size = (*item.config)[ConfigParams::DNN_FC_OUTPUT_SIZE].As<decltype(common_config_[item.id_group].dnn_fc_output_size)>(common_config_[item.id_group].dnn_fc_output_size);

        common_config_[item.id_group].dnn_fr_model_name = (*item.config)[ConfigParams::DNN_FR_MODEL_NAME].As<decltype(common_config_[item.id_group].dnn_fr_model_name)>(common_config_[item.id_group].dnn_fr_model_name);
        common_config_[item.id_group].dnn_fr_input_width = (*item.config)[ConfigParams::DNN_FR_INPUT_WIDTH].As<decltype(common_config_[item.id_group].dnn_fr_input_width)>(common_config_[item.id_group].dnn_fr_input_width);
        common_config_[item.id_group].dnn_fr_input_height = (*item.config)[ConfigParams::DNN_FR_INPUT_HEIGHT].As<decltype(common_config_[item.id_group].dnn_fr_input_height)>(common_config_[item.id_group].dnn_fr_input_height);
        common_config_[item.id_group].dnn_fr_input_tensor_name = (*item.config)[ConfigParams::DNN_FR_INPUT_TENSOR_NAME].As<decltype(common_config_[item.id_group].dnn_fr_input_tensor_name)>(common_config_[item.id_group].dnn_fr_input_tensor_name);
        common_config_[item.id_group].dnn_fr_output_tensor_name = (*item.config)[ConfigParams::DNN_FR_OUTPUT_TENSOR_NAME].As<decltype(common_config_[item.id_group].dnn_fr_output_tensor_name)>(common_config_[item.id_group].dnn_fr_output_tensor_name);
        common_config_[item.id_group].dnn_fr_output_size = (*item.config)[ConfigParams::DNN_FR_OUTPUT_SIZE].As<decltype(common_config_[item.id_group].dnn_fr_output_size)>(common_config_[item.id_group].dnn_fr_output_size);

        common_config_[item.id_group].comments_blurry_face = (*item.config)[ConfigParams::COMMENTS_BLURRY_FACE].As<decltype(common_config_[item.id_group].comments_blurry_face)>(common_config_[item.id_group].comments_blurry_face);
        common_config_[item.id_group].comments_descriptor_creation_error = (*item.config)[ConfigParams::COMMENTS_DESCRIPTOR_CREATION_ERROR].As<decltype(common_config_[item.id_group].comments_descriptor_creation_error)>(common_config_[item.id_group].comments_descriptor_creation_error);
        common_config_[item.id_group].comments_descriptor_exists = (*item.config)[ConfigParams::COMMENTS_DESCRIPTOR_EXISTS].As<decltype(common_config_[item.id_group].comments_descriptor_exists)>(common_config_[item.id_group].comments_descriptor_exists);
        common_config_[item.id_group].comments_inference_error = (*item.config)[ConfigParams::COMMENTS_INFERENCE_ERROR].As<decltype(common_config_[item.id_group].comments_inference_error)>(common_config_[item.id_group].comments_inference_error);
        common_config_[item.id_group].comments_new_descriptor = (*item.config)[ConfigParams::COMMENTS_NEW_DESCRIPTOR].As<decltype(common_config_[item.id_group].comments_new_descriptor)>(common_config_[item.id_group].comments_new_descriptor);
        common_config_[item.id_group].comments_no_faces = (*item.config)[ConfigParams::COMMENTS_NO_FACES].As<decltype(common_config_[item.id_group].comments_no_faces)>(common_config_[item.id_group].comments_no_faces);
        common_config_[item.id_group].comments_non_frontal_face = (*item.config)[ConfigParams::COMMENTS_NON_FRONTAL_FACE].As<decltype(common_config_[item.id_group].comments_non_frontal_face)>(common_config_[item.id_group].comments_non_frontal_face);
        common_config_[item.id_group].comments_non_normal_face_class = (*item.config)[ConfigParams::COMMENTS_NON_NORMAL_FACE_CLASS].As<decltype(common_config_[item.id_group].comments_non_normal_face_class)>(common_config_[item.id_group].comments_non_normal_face_class);
        common_config_[item.id_group].comments_partial_face = (*item.config)[ConfigParams::COMMENTS_PARTIAL_FACE].As<decltype(common_config_[item.id_group].comments_partial_face)>(common_config_[item.id_group].comments_partial_face);
        common_config_[item.id_group].comments_url_image_error = (*item.config)[ConfigParams::COMMENTS_URL_IMAGE_ERROR].As<decltype(common_config_[item.id_group].comments_url_image_error)>(common_config_[item.id_group].comments_url_image_error);

        common_config_[item.id_group].sg_max_descriptor_count = (*item.config)[ConfigParams::SG_MAX_DESCRIPTOR_COUNT].As<decltype(common_config_[item.id_group].sg_max_descriptor_count)>(common_config_[item.id_group].sg_max_descriptor_count);

        // default video stream config
        default_vstream_config_[item.id_group] = updateVStreamConfig(*item.config);
        default_vstream_config_[item.id_group].id_group = item.id_group;
      }
    }

    static size_t size()
    {
      return 0;
    }

    [[nodiscard]] const auto& getCommonConfig() const
    {
      return common_config_;
    }

    [[nodiscard]] const auto& getDefaultVStreamConfig() const
    {
      return default_vstream_config_;
    }

  private:
    HashMap<int32_t, CommonConfig> common_config_;
    HashMap<int32_t, VStreamConfig> default_vstream_config_{};
  };

  struct ConfigCachePolicy
  {
    static constexpr std::string_view kName = "frs-config-pg-cache";

    using ValueType = ConfigData;
    static constexpr auto kKeyMember = &ConfigData::id_group;
    static constexpr auto kQuery = R"__SQL__(
      select
        vg.id_group,
        coalesce(cc.config, '{}') || coalesce(dvc.config, '{}') config
      from
        vstream_groups vg
        left join common_config cc
          on cc.id_group = vg.id_group
        left join default_vstream_config dvc
          on dvc.id_group = vg.id_group
    )__SQL__";
    static constexpr const char* kUpdatedField = nullptr;
    using CacheContainer = ConfigContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using ConfigCache = userver::components::PostgreCache<ConfigCachePolicy>;

  // Video streams configuration cache
  struct VStreamConfigData
  {
    std::string unique_key;
    int32_t id_group{};
    int32_t id_vstream{};
    std::string vstream_ext;
    std::string url;
    std::string callback_url;
    std::optional<userver::formats::json::Value> config;
    bool flag_deleted{};
  };

  class VStreamConfigContainer
  {
  public:
    void insert_or_assign(const std::string& key, VStreamConfigData&& item)
    {
      if (item.flag_deleted)
      {
        // item is mark for deletion, so remove it from cache
        data_.erase(key);
      } else
      {
        // save data in cache
        VStreamConfig config{};
        if (item.config)
          config = updateVStreamConfig(*item.config);
        config.id_group = item.id_group;
        config.id_vstream = item.id_vstream;
        config.vstream_ext = item.vstream_ext;
        config.url = item.url;
        config.callback_url = item.callback_url;
        data_[key] = std::move(config);
      }
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
    static constexpr std::string_view kName = "frs-vstreams-config-pg-cache";

    using ValueType = VStreamConfigData;
    static constexpr auto kKeyMember = &VStreamConfigData::unique_key;
    static constexpr auto kQuery = R"__SQL__(
      select
        concat(vs.id_group, '_', vs.vstream_ext) unique_key,
        vs.id_group,
        vs.id_vstream,
        vs.vstream_ext,
        coalesce(vs.url, '') url,
        coalesce(vs.callback_url, '') callback_url,
        coalesce(d.config, '{}') || coalesce(vs.config, '{}') config,
        vs.flag_deleted
      from
        video_streams vs
        left join default_vstream_config d
          on d.id_group = vs.id_group
    )__SQL__";
    static constexpr auto kUpdatedField = "last_updated";
    using UpdatedFieldType = userver::storages::postgres::TimePointTz;
    using CacheContainer = VStreamConfigContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using VStreamsConfigCache = userver::components::PostgreCache<VStreamsConfigCachePolicy>;

  // Face descriptors
  struct FaceDescriptorData
  {
    int32_t id_descriptor{};
    userver::storages::postgres::ByteaWrapper<std::string> descriptor_data;
    bool flag_deleted{};
  };

  // Custom cache container which inserts and erases data in insert_or_assign function.
  // The base idea is to mark data for deletion in database and remove it later by scheduler task.
  class FaceDescriptorCacheContainer
  {
  public:
    void insert_or_assign(const int32_t id_descriptor, FaceDescriptorData&& item)
    {
      if (item.flag_deleted)
      {
        // item is mark for deletion, so remove it from cache
        data_.erase(id_descriptor);
      } else
      {
        // save data in cache
        FaceDescriptor fd;
        fd.create(1, static_cast<int>(item.descriptor_data.bytes.size() / sizeof(float)), CV_32F);
        std::memmove(fd.data, item.descriptor_data.bytes.data(), item.descriptor_data.bytes.size());
        double norm_l2 = cv::norm(fd, cv::NORM_L2);
        if (norm_l2 <= 0.0)
          norm_l2 = 1.0;
        fd = fd / norm_l2;
        data_[id_descriptor] = std::move(fd);
      }
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
    HashMap<int32_t, FaceDescriptor> data_;
  };

  struct FaceDescriptorPolicy
  {
    static constexpr std::string_view kName = "frs-face-descriptor-pg-cache";

    using ValueType = FaceDescriptorData;
    static constexpr auto kKeyMember = &FaceDescriptorData::id_descriptor;
    static constexpr auto kQuery = "select id_descriptor, descriptor_data, flag_deleted from face_descriptors";
    static constexpr auto kUpdatedField = "last_updated";
    using UpdatedFieldType = userver::storages::postgres::TimePointTz;
    using CacheContainer = FaceDescriptorCacheContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using FaceDescriptorCache = userver::components::PostgreCache<FaceDescriptorPolicy>;

  // Video streams and bound faces cache
  struct VStreamDescriptors
  {
    std::string unique_key;
    int32_t id_vstream;
    int32_t id_descriptor;
    bool flag_deleted;
  };

  // Custom cache container which inserts and erases data in insert_or_assign function.
  // The base idea is to mark data for deletion in database and remove it later by scheduler task.
  class VStreamDescriptorsCacheContainer
  {
  public:
    void insert_or_assign(const std::string&, VStreamDescriptors&& item)
    {
      if (item.flag_deleted)
      {
        // item is mark for deletion, so remove it from cache
        if (data_.contains(item.id_vstream))
          data_[item.id_vstream].erase(item.id_descriptor);
      } else
        // save data in cache
        data_[item.id_vstream].insert(item.id_descriptor);
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
    HashMap<int32_t, HashSet<int32_t>> data_;
  };

  struct VStreamDescriptorsPolicy
  {
    static constexpr std::string_view kName = "frs-vstream-descriptors-pg-cache";

    using ValueType = VStreamDescriptors;
    static constexpr auto kKeyMember = &VStreamDescriptors::unique_key;
    static constexpr auto kQuery = "select concat(id_vstream, '_', id_descriptor) unique_key, id_vstream, id_descriptor, flag_deleted from link_descriptor_vstream";
    static constexpr auto kUpdatedField = "last_updated";
    using UpdatedFieldType = userver::storages::postgres::TimePointTz;
    using CacheContainer = VStreamDescriptorsCacheContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using VStreamDescriptorsCache = userver::components::PostgreCache<VStreamDescriptorsPolicy>;

  // Special groups configuration cache
  struct SGConfig
  {
    std::string sg_api_token;
    int32_t id_special_group{};
    std::string callback_url;
    int32_t max_descriptor_count{};
    int32_t id_group{};
  };

  class SGConfigCacheContainer
  {
  public:
    void insert_or_assign(const std::string& key, SGConfig&& item)
    {
      map_id_[item.id_special_group] = key;
      map_id_group_[item.id_group].insert(item.id_special_group);
      data_[key] = std::move(item);
    }

    static size_t size()
    {
      return 0;
    }

    [[nodiscard]] const auto& getData() const
    {
      return data_;
    }

    [[nodiscard]] const auto& getMap() const
    {
      return map_id_;
    }

    [[nodiscard]] const auto& getMappedSG() const
    {
      return map_id_group_;
    }

  private:
    HashMap<std::string, SGConfig> data_;
    HashMap<int32_t, std::string> map_id_;
    HashMap<int32_t, HashSet<int32_t>> map_id_group_;
  };

  struct SGConfigCachePolicy
  {
    static constexpr std::string_view kName = "frs-sg-config-pg-cache";

    using ValueType = SGConfig;
    static constexpr auto kKeyMember = &SGConfig::sg_api_token;
    static constexpr auto kQuery = "select sg_api_token, id_special_group, coalesce(callback_url, '') callback_url, max_descriptor_count, id_group from special_groups";
    static constexpr const char* kUpdatedField = nullptr;
    using CacheContainer = SGConfigCacheContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using SGConfigCache = userver::components::PostgreCache<SGConfigCachePolicy>;

  // Special groups and bound faces cache
  struct SGDescriptors
  {
    std::string unique_key;
    int32_t id_sgroup;
    int32_t id_descriptor;
    bool flag_deleted;
  };

  class SGDescriptorsCacheContainer
  {
  public:
    void insert_or_assign(const std::string&, SGDescriptors&& item)
    {
      if (item.flag_deleted)
      {
        // item is mark for deletion, so remove it from cache
        if (data_.contains(item.id_sgroup))
          data_[item.id_sgroup].erase(item.id_descriptor);
      } else
        // save data in cache
        data_[item.id_sgroup].insert(item.id_descriptor);
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
    HashMap<int32_t, HashSet<int32_t>> data_;
  };

  struct SGDescriptorsPolicy
  {
    static constexpr std::string_view kName = "frs-sg-descriptors-pg-cache";

    using ValueType = SGDescriptors;
    static constexpr auto kKeyMember = &SGDescriptors::unique_key;
    static constexpr auto kQuery = "select concat(id_sgroup, '_', id_descriptor) unique_key, id_sgroup, id_descriptor, flag_deleted from link_descriptor_sgroup";
    static constexpr auto kUpdatedField = "last_updated";
    using UpdatedFieldType = userver::storages::postgres::TimePointTz;
    using CacheContainer = SGDescriptorsCacheContainer;
    static constexpr auto kClusterHostType = userver::storages::postgres::ClusterHostType::kSlave;
  };

  using SGDescriptorsCache = userver::components::PostgreCache<SGDescriptorsPolicy>;

}  // namespace Frs
