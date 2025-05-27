#pragma once

#include <absl/strings/string_view.h>
#include <userver/clients/http/component.hpp>
#include <userver/concurrent/background_task_storage.hpp>
#include <userver/concurrent/variable.hpp>
#include <userver/logging/component.hpp>

#include "lprs_caches.hpp"

namespace Lprs
{
  // License plate classes
  inline static constexpr char const* PLATE_CLASSES[] = {"ru_1", "ru_1a"};
  inline static constexpr int32_t PLATE_CLASS_COUNT = 2;  // must be equal to the length of PLATE_CLASSES array
  inline static constexpr int32_t PLATE_CLASS_RU_1 = 0;
  inline static constexpr int32_t PLATE_CLASS_RU_1A = 1;

  namespace DatabaseFields
  {
    inline static constexpr auto ID_VSTREAM = "id_vstream";
    inline static constexpr auto LOG_DATE = "log_date";
    inline static constexpr auto INFO = "info";
    inline static constexpr auto CONFIG = "config";
  }  // namespace DatabaseFields

  struct LocalConfig
  {
    int32_t allow_group_id_without_auth{1};
    std::chrono::milliseconds ban_maintenance_interval{std::chrono::seconds{5}};
    std::chrono::milliseconds events_log_maintenance_interval{std::chrono::hours{2}};
    std::chrono::milliseconds events_log_ttl{std::chrono::hours{4}};
    std::string events_screenshots_path;
    std::string events_screenshots_url_prefix;
    std::string failed_path;
    std::chrono::milliseconds failed_ttl{std::chrono::days{60}};
  };

  struct PlateNumberData
  {
    std::string number;
    float score;
  };

  struct LicensePlate
  {
    float bbox[4];  // absolute xmin, ymin, xmax, ymax
    float confidence;
    float kpts[8]{};  // four key points
    int32_t plate_class;
    std::vector<PlateNumberData> plate_numbers;
  };

  struct CharData
  {
    float bbox[4];  // plate relative xmin, ymin, xmax, ymax
    float confidence;
    int32_t char_class;
    int32_t plate_class;
  };

  struct BannedPlateData
  {
    std::chrono::time_point<std::chrono::steady_clock> tp1;
    std::chrono::time_point<std::chrono::steady_clock> tp2;
    cv::Rect2f bbox;
  };

  struct Vehicle
  {
    float bbox[4];  // absolute xmin, ymin, xmax, ymax
    float confidence;
    bool is_special;
    std::vector<LicensePlate> license_plates;
  };

  class Workflow final : public userver::components::LoggableComponentBase
  {
  public:
    static constexpr std::string_view kName = "lprs-workflow";
    static constexpr std::string_view kDatabase = "lprs-postgresql-database";
    static constexpr std::string_view kLogger = "lprs";
    std::string kBanMaintenanceName = "ban_maintenance";
    std::string kEventsLogMaintenanceName = "events_log_maintenance";

    // queries
    static constexpr auto SQL_ADD_EVENT = R"__SQL__(
      insert into events_log(id_vstream, log_date, info) values($1, $2, $3) returning id_event;
    )__SQL__";

    inline static constexpr auto SQL_REMOVE_OLD_EVENTS = R"__SQL__(
      delete from events_log where log_date < $1;
    )__SQL__";

    Workflow(const userver::components::ComponentConfig& config,
      const userver::components::ComponentContext& context);
    static userver::yaml_config::Schema GetStaticConfigSchema();

    void startWorkflow(std::string&& vstream_key);
    void stopWorkflow(std::string&& vstream_key, bool is_internal = true);
    const LocalConfig& getLocalConfig();
    const userver::logging::LoggerPtr& getLogger();

  private:
    userver::concurrent::BackgroundTaskStorageCore tasks_;
    userver::engine::TaskProcessor& task_processor_;
    userver::engine::TaskProcessor& fs_task_processor_;
    userver::clients::http::Client& http_client_;
    const VStreamsConfigCache& vstreams_config_cache_;
    userver::storages::postgres::ClusterPtr pg_cluster_;
    userver::utils::PeriodicTask ban_maintenance_task_;
    userver::utils::PeriodicTask events_log_maintenance_task_;
    userver::logging::LoggerPtr logger_;

    LocalConfig local_config_;

    userver::concurrent::Variable<HashMap<std::string, bool>> being_processed_vstreams;
    userver::concurrent::Variable<HashMap<std::string, BannedPlateData>> ban_data;
    userver::concurrent::Variable<HashMap<std::string, std::chrono::time_point<std::chrono::steady_clock>>> ban_special_data;
    userver::concurrent::Variable<HashMap<std::string, std::chrono::time_point<std::chrono::steady_clock>>> vstream_timeouts;

    void OnAllComponentsAreStopping() override;
    void processPipeline(std::string&& vstream_key);
    void doBanMaintenance();
    void doEventsLogMaintenance() const;
    void nextPipeline(std::string&& vstream_key, std::chrono::milliseconds delay);

    // Inference pipeline methods
    static std::vector<float> preprocessImageForVdNet(const cv::Mat& img, int32_t width, int32_t height, cv::Point2f& shift, double& scale);
    static std::vector<float> preprocessImageForVcNet(const cv::Mat& img, int32_t width, int32_t height);
    static std::vector<float> preprocessImageForLpdNet(const cv::Mat& img, int32_t width, int32_t height, cv::Point2f& shift, double& scale);
    static std::vector<float> preprocessImageForLprNet(const cv::Mat& img, int32_t width, int32_t height, cv::Point2f& shift, double& scale);

    // VDNet
    bool doInferenceVdNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles) const;

    // VCNet
    bool doInferenceVcNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles) const;

    // LPDNet
    bool doInferenceLpdNet(const cv::Mat& img, const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles);
    void removeDuplicatePlates(const VStreamConfig& config, std::vector<Vehicle>& detected_vehicles, int32_t width, int32_t height) const;

    // LPRNet
    bool doInferenceLprNet(const cv::Mat& img, const VStreamConfig& config, std::vector<LicensePlate*>& detected_plates);

    static bool isValidPlateNumber(absl::string_view plate_number, int32_t plate_class);
    int64_t addEventLog(int32_t id_vstream, const userver::storages::postgres::TimePointTz& log_date, const userver::formats::json::Value& info) const;
  };
}  // namespace Lprs
