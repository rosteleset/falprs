#pragma once

#include <absl/strings/string_view.h>
#include <userver/server/handlers/http_handler_json_base.hpp>
#include <userver/storages/postgres/cluster.hpp>
#include <userver/storages/postgres/component.hpp>

#include "lprs_caches.hpp"
#include "lprs_workflow.hpp"

namespace Lprs
{
  class Api final : public userver::server::handlers::HttpHandlerJsonBase
  {
  public:
    static constexpr auto kName = "lprs-api-http";

    // API methods
    static constexpr auto METHOD_ADD_STREAM = "addStream";                              // add or change video stream
    static constexpr auto METHOD_REMOVE_STREAM = "removeStream";                        // remove video stream
    static constexpr auto METHOD_LIST_STREAMS = "listStreams";                          // list video streams with config
    static constexpr auto METHOD_START_WORKFLOW = "startWorkflow";                      // start license plate recognition workflow
    static constexpr auto METHOD_STOP_WORKFLOW = "stopWorkflow";                        // stop license plate recognition workflow
    static constexpr auto METHOD_GET_EVENT_DATA = "getEventData";                       // get a list of events from a time interval
    static constexpr auto METHOD_SET_STREAM_DEFAULT_CONFIG = "setStreamDefaultConfig";  // set the default video stream configuration parameters
    static constexpr auto METHOD_GET_STREAM_DEFAULT_CONFIG = "getStreamDefaultConfig";  // get the default video stream configuration parameters

    // parameters
    static constexpr auto PARAM_STREAM_ID = "streamId";
    static constexpr auto PARAM_CONFIG = "config";
    static constexpr auto PARAM_CODE = "code";
    static constexpr auto PARAM_MESSAGE = "message";
    static constexpr auto PARAM_DATA = "data";
    static constexpr auto PARAM_SCREENSHOT_URL = "screenshotUrl";
    static constexpr auto PARAM_EVENT_ID = "eventId";
    static constexpr auto PARAM_EVENT_DATE = "date";
    static constexpr auto PARAM_VEHICLES_INFO = "vehicles";
    static constexpr auto PARAM_IS_SPECIAL = "isSpecial";
    static constexpr auto PARAM_HAS_SPECIAL = "hasSpecial";
    static constexpr auto PARAM_PLATES_INFO = "plates";
    static constexpr auto PARAM_BOX = "box";
    static constexpr auto PARAM_KPTS = "kpts";
    static constexpr auto PARAM_NUMBER = "number";
    static constexpr auto PARAM_CONFIDENCE = "confidence";
    static constexpr auto PARAM_SCORE = "score";
    static constexpr auto PARAM_PLATE_TYPE = "type";

    // messages
    static constexpr auto MESSAGE_OK = "Ok";

    // errors
    static constexpr auto ERROR_NO_METHOD = "Method not found";

    // queries
    static constexpr auto SQL_ADD_STREAM = R"__SQL__(
      insert into vstreams(id_group, ext_id, config) values($1, $2, $3)
    )__SQL__";

    static constexpr auto SQL_GET_STREAM = R"__SQL__(
      select
        id_vstream
      from
        vstreams
      where
        id_group = $1
        and ext_id = $2
    )__SQL__";

    static constexpr auto SQL_UPDATE_STREAM = R"__SQL__(
      update
        vstreams
      set
        config = $1
      where
        id_vstream = $2
    )__SQL__";

    static constexpr auto SQL_REMOVE_STREAM = R"__SQL__(
      delete from
        vstreams
      where
        id_group = $1
        and ext_id = $2
    )__SQL__";

    static constexpr auto SQL_LIST_STREAMS = R"__SQL__(
      select
        ext_id,
        config
      from
        vstreams
      where
        id_group = $1
      order by
        ext_id
    )__SQL__";

    static constexpr auto SQL_GET_EVENT_BY_ID = R"__SQL__(
      select
        log_date,
        info,
        id_vstream
      from
        events_log
      where
        id_event = $1
    )__SQL__";

    static constexpr auto SQL_GET_NEAREST_EVENT = R"__SQL__(
      select
        log_date,
        info
      from
        events_log
      where
        id_vstream = $0
        and (log_date > timestamptz '$1' - interval '$2 millisecond')
        and (log_date < timestamptz '$1' + interval '$3 millisecond')
      order by
        abs(extract(epoch from (log_date - timestamptz '$1')))
      limit
        1
    )__SQL__";

    static constexpr auto SQL_SET_STREAM_DEFAULT_CONFIG_PARAMS = "update default_vstream_config set config = coalesce(config, $2) || $2 where id_group = $1";
    static constexpr auto SQL_GET_STREAM_DEFAULT_CONFIG_PARAMS = "select config from default_vstream_config where id_group = $1";

    // Component is valid after construction and is able to accept requests
    Api(const userver::components::ComponentConfig& config, const userver::components::ComponentContext& context);

    userver::formats::json::Value HandleRequestJsonThrow(const userver::server::http::HttpRequest&,
      const userver::formats::json::Value& json,
      userver::server::request::RequestContext&) const override;

  private:
    const GroupsCache& groups_cache_;
    userver::storages::postgres::ClusterPtr pg_cluster_;
    Workflow& workflow_;
    const VStreamsConfigCache& vstreams_config_cache_;
    const VStreamGroupCache& vstream_group_cache_;

    int32_t checkToken(absl::string_view token) const;
    static void requireMemberThrow(const userver::formats::json::Value& json, absl::string_view member);
    void addStream(int32_t id_group, const userver::formats::json::Value& json) const;
    void removeStream(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value listStreams(int32_t id_group) const;
    void startWorkflow(int32_t id_group, const userver::formats::json::Value& json) const;
    void stopWorkflow(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value getEventData(int32_t id_group, const userver::formats::json::Value& json) const;
    void setStreamDefaultConfigParams(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value getStreamDefaultConfigParams(int32_t id_group) const;
  };
}  // namespace Lprs
