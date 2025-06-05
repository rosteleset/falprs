#pragma once

#include <absl/strings/str_replace.h>
#include <userver/clients/http/client.hpp>
#include <userver/components/loggable_component_base.hpp>
#include <userver/concurrent/background_task_storage.hpp>
#include <userver/logging/component.hpp>
#include <userver/storages/postgres/postgres_fwd.hpp>

#include "frs_caches.hpp"

namespace Frs
{
  namespace DatabaseFields
  {
    inline static constexpr auto ID_GROUP = "id_group";
    inline static constexpr auto ID_VSTREAM = "id_vstream";
    inline static constexpr auto VSTREAM_EXT = "vstream_ext";
    inline static constexpr auto FACE_LEFT = "face_left";
    inline static constexpr auto FACE_TOP = "face_top";
    inline static constexpr auto FACE_WIDTH = "face_width";
    inline static constexpr auto FACE_HEIGHT = "face_height";
    inline static constexpr auto FACE_IMAGE = "face_image";
    inline static constexpr auto ID_LOG = "id_log";
    inline static constexpr auto LOG_DATE = "log_date";
    inline static constexpr auto SCREENSHOT_URL = "screenshot_url";
    inline static constexpr auto COPY_EVENT_DATA = "copy_data";
    inline static constexpr auto ID_DESCRIPTOR = "id_descriptor";
    inline static constexpr auto DESCRIPTOR_DATA = "descriptor_data";
    inline static constexpr auto QUALITY = "quality";
    inline static constexpr auto EXT_EVENT_UUID = "ext_event_uuid";
    inline static constexpr auto URL = "url";
    inline static constexpr auto CALLBACK_URL = "callback_url";
    inline static constexpr auto ID_SPECIAL_GROUP = "id_special_group";
    inline static constexpr auto LOG_UUID = "log_uuid";
    inline static constexpr auto CONFIG = "config";
    inline static constexpr auto SG_API_TOKEN = "sg_api_token";
    inline static constexpr auto SG_NAME = "group_name";
    inline static constexpr auto SG_MAX_DESCRIPTOR_COUNT = "max_descriptor_count";
  }

  enum CopyEventData
  {
    DISABLED = -1,
    NONE = 0,
    SCHEDULED = 1
  };

  struct LocalConfig
  {
    int32_t allow_group_id_without_auth{1};
    std::string events_path;
    std::string screenshots_path;
    std::string screenshots_url_prefix;
    std::chrono::milliseconds clear_old_log_faces{std::chrono::hours{1}};
    std::chrono::milliseconds flag_deleted_maintenance_interval{std::chrono::seconds{10}};
    std::chrono::milliseconds flag_deleted_ttl{std::chrono::minutes{5}};
    std::chrono::milliseconds copy_events_maintenance_interval{std::chrono::seconds{30}};
    std::chrono::milliseconds clear_old_events{std::chrono::days{1}};
    std::chrono::milliseconds log_faces_ttl{std::chrono::hours{4}};
    std::chrono::milliseconds events_ttl{std::chrono::days{30}};
  };

  enum TaskType
  {
    TASK_NONE,
    TASK_RECOGNIZE,
    TASK_REGISTER_DESCRIPTOR,
    TASK_PROCESS_FRAME,
    TASK_TEST
  };

  struct TaskData
  {
    int32_t id_group{};
    std::string vstream_key;
    TaskType task_type{TASK_NONE};
    std::string frame_url;
    int face_left{};
    int face_top{};
    int face_width{};
    int face_height{};
    int id_sgroup{};
  };

  struct DescriptorRegistrationResult
  {
    int32_t id_descriptor{};
    std::string comments;
    cv::Mat face_image{};
    int face_left{};
    int face_top{};
    int face_width{};
    int face_height{};
    std::vector<int> id_descriptors{};
  };

  // to collect inference statistics
  struct DNNStatsData
  {
    int fd_count{};
    int fc_count{};
    int fr_count{};
  };

  struct alignas(float) FaceDetection
  {
    float bbox[4];  // x1 y1 x2 y2
    float face_confidence;
    float landmark[10];
  };

  enum FaceClassIndexes
  {
    FACE_NONE = -1,
    FACE_NORMAL = 0,
  };

  enum DeliveryEventResult
  {
    ERROR = 0,
    SUCCESSFUL = 1
  };

  struct SGroupFaceData
  {
    double cosine_distance = -2.0;
    int id_descriptor = 0;
  };

  struct FaceData
  {
    cv::Rect face_rect;
    bool is_work_area = false;
    bool is_frontal = false;
    bool is_non_blurry = false;
    FaceClassIndexes face_class_index = FACE_NONE;
    float face_class_confidence = 0.0f;
    double cosine_distance = -2.0;
    FaceDescriptor fd;
    cv::Mat landmarks5;
    double laplacian = 0.0;
    double ioa = 0.0;
    int id_descriptor = 0;
    HashMap<int, SGroupFaceData> sg_descriptors;
  };

  struct FaceClass
  {
    int class_index;
    float score;
  };

  struct UnknownDescriptorData
  {
    std::chrono::time_point<std::chrono::steady_clock> expiration_tp;
    FaceDescriptor fd;
    cv::Mat face_image;
  };

  class Workflow final : public userver::components::LoggableComponentBase
  {
  public:
    static constexpr std::string_view kName = "frs-workflow";
    static constexpr std::string_view kDatabase = "frs-postgresql-database";
    static constexpr std::string_view kLogger = "frs";
    std::string kOldLogsMaintenance = "old_logs_maintenance";
    std::string kFlagDeletedMaintenance = "flag_deleted_maintenance";
    std::string kCopyEventsMaintenance = "copy_events_maintenance";
    std::string kOldEventsMaintenance = "old_events_maintenance";

    static constexpr std::string_view MIME_IMAGE = "image/jpeg";
    static constexpr std::string_view DATE_FORMAT = "%Y-%m-%d";
    static constexpr std::string_view DATA_FILE_SUFFIX = ".dat";
    static constexpr std::string_view JSON_SUFFIX = ".json";

    // SQL queries
    static constexpr auto SQL_ADD_LOG_FACE = R"__SQL__(
      insert into log_faces(id_vstream, log_date, id_descriptor, quality, face_left, face_top, face_width, face_height, screenshot_url, log_uuid, copy_data)
      values($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) returning id_log
    )__SQL__";

    static constexpr auto SQL_ADD_FACE_DESCRIPTOR = R"_SQL_(
      insert into face_descriptors(id_group, descriptor_data, id_parent) values($1, $2, $3) returning id_descriptor
    )_SQL_";

    static constexpr auto SQL_ADD_DESCRIPTOR_IMAGE = R"_SQL_(
      insert into descriptor_images(id_descriptor, mime_type, face_image) values($1, $2, $3)
    )_SQL_";

    static constexpr auto SQL_ADD_LINK_DESCRIPTOR_SGROUP = R"_SQL_(
      insert into link_descriptor_sgroup(id_sgroup, id_descriptor) values($1, $2)
    )_SQL_";

    static constexpr auto SQL_REMOVE_OLD_LOG_FACES = R"_SQL_(
      delete from
        log_faces
      where
        log_date < $1
    )_SQL_";

    static constexpr auto SQL_DELETE_VIDEO_STREAMS_MARKED = R"_SQL_(
      delete from
        video_streams
      where
        flag_deleted
        and last_updated < $1
    )_SQL_";

    static constexpr auto SQL_DELETE_FACE_DESCRIPTORS_MARKED = R"_SQL_(
      delete from
        face_descriptors
      where
        flag_deleted
        and last_updated < $1
    )_SQL_";

    static constexpr auto SQL_DELETE_LINK_DESCRIPTOR_VSTREAM_MARKED = R"_SQL_(
      delete from
        link_descriptor_vstream
      where
        flag_deleted
        and last_updated < $1
    )_SQL_";

    static constexpr auto SQL_DELETE_SPECIAL_GROUPS_MARKED = R"_SQL_(
      delete from
        special_groups
      where
        flag_deleted
        and last_updated < $1
    )_SQL_";

    static constexpr auto SQL_DELETE_LINK_DESCRIPTOR_SGROUP_MARKED = R"_SQL_(
      delete from
        link_descriptor_sgroup
      where
        flag_deleted
        and last_updated < $1
    )_SQL_";

    static constexpr auto SQL_GET_LOG_COPY_DATA = R"_SQL_(
      select
        l.id_log,
        v.id_group,
        l.id_vstream,
        l.log_uuid::varchar,
        l.ext_event_uuid,
        l.log_date
      from
        log_faces l
        inner join video_streams v
          on v.id_vstream = l.id_vstream
      where
        l.copy_data = 1
    )_SQL_";

    static constexpr auto SQL_UPDATE_LOG_COPY_DATA = "update log_faces set copy_data = 2 where id_log = $1";

    Workflow(const userver::components::ComponentConfig& config,
      const userver::components::ComponentContext& context);
    ~Workflow() override;
    static userver::yaml_config::Schema GetStaticConfigSchema();
    [[nodiscard]] const userver::logging::LoggerPtr& getLogger() const;
    [[nodiscard]] const LocalConfig& getLocalConfig() const;
    void startWorkflow(std::string&& vstream_key);
    void stopWorkflow(std::string&& vstream_key, bool is_internal = true);
    DescriptorRegistrationResult processPipeline(TaskData&& task_data);
    void loadDNNStatsData();
    void saveDNNStatsData() const;

  private:
    userver::concurrent::BackgroundTaskStorageCore tasks_;
    userver::engine::TaskProcessor& task_processor_;
    userver::engine::TaskProcessor& fs_task_processor_;
    userver::clients::http::Client& http_client_;
    userver::logging::LoggerPtr logger_;
    userver::storages::postgres::ClusterPtr pg_cluster_;
    const ConfigCache& common_config_cache_;
    const VStreamsConfigCache& vstreams_config_cache_;
    const FaceDescriptorCache& face_descriptor_cache_;
    const VStreamDescriptorsCache& vstream_descriptors_cache_;
    const SGConfigCache& sg_config_cache_;
    const SGDescriptorsCache& sg_descriptors_cache_;
    userver::utils::PeriodicTask old_logs_maintenance_task_;
    userver::utils::PeriodicTask flag_deleted_maintenance_task_;
    userver::utils::PeriodicTask copy_events_maintenance_task_;
    userver::utils::PeriodicTask old_events_maintenance_task_;

    LocalConfig local_config_;

    userver::concurrent::Variable<HashMap<std::string, bool>> being_processed_vstreams;
    userver::concurrent::Variable<HashMap<int32_t, DNNStatsData>> dnn_stats_data;
    userver::concurrent::Variable<HashMap<std::string, std::chrono::time_point<std::chrono::steady_clock>>> vstream_timeouts;
    userver::concurrent::Variable<HashMap<int32_t, std::vector<UnknownDescriptorData>>> unknown_descriptors;

    // Maintenance member functions
    void doOldLogMaintenance() const;
    void doFlagDeletedMaintenance() const;
    void doCopyEventsMaintenance() const;
    void doOldEventsMaintenance() const;

    void nextPipeline(TaskData&& task_data, std::chrono::milliseconds delay);

    // Inference pipeline functions
    static cv::Mat preprocessImage(const cv::Mat& img, int width, int height, float& scale);
    bool detectFaces(const TaskData& task_data, const cv::Mat& frame, const VStreamConfig& config,
      std::vector<FaceDetection>& detected_faces);
    bool inferFaceClass(const TaskData& task_data, const cv::Mat& aligned_face, const VStreamConfig& config,
      std::vector<FaceClass>& face_classes);
    bool extractFaceDescriptor(const TaskData& task_data, const cv::Mat& aligned_face, const VStreamConfig& config,
      FaceDescriptor& face_descriptor);
    int64_t addLogFace(int32_t id_vstream, const userver::storages::postgres::TimePointTz& log_date,
      int32_t id_descriptor, double quality, const cv::Rect& face_rect, const std::string& screenshot_url, const boost::uuids::uuid& uuid, CopyEventData copy_event_data = NONE) const;
    int32_t addFaceDescriptor(int32_t id_group, int32_t id_vstream, const FaceDescriptor& fd, const cv::Mat& f_img, int32_t id_parent = 0);
    int32_t addSGroupFaceDescriptor(int32_t id_sgroup, const FaceDescriptor& fd, const cv::Mat& f_img);
  };
}  // namespace Frs
