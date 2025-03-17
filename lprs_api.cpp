#include <string>

#include <absl/strings/substitute.h>
#include <userver/components/component_context.hpp>

#include "lprs_api.hpp"

namespace Lprs
{
  Api::Api(const userver::components::ComponentConfig& config,
    const userver::components::ComponentContext& context)
    : HttpHandlerJsonBase(config, context),
      groups_cache_(context.FindComponent<GroupsCache>()),
      pg_cluster_(context.FindComponent<userver::components::Postgres>(Workflow::kDatabase).GetCluster()),
      workflow_(context.FindComponent<Workflow>()),
      vstreams_config_cache_(context.FindComponent<VStreamsConfigCache>()),
      vstream_group_cache_(context.FindComponent<VStreamGroupCache>())
  {
  }

  userver::formats::json::Value Api::HandleRequestJsonThrow(const userver::server::http::HttpRequest& request,
    const userver::formats::json::Value& json,
    userver::server::request::RequestContext&) const
  {
    // check authorization
    const auto& auth_value = request.GetHeader("Authorization");
    const auto& api_method = request.GetPathArg(0);
    LOG_INFO_TO(workflow_.getLogger()) << "API call " << api_method << ": " << json;
    int32_t id_group = workflow_.getLocalConfig().allow_group_id_without_auth;

    // check authorization
    if (auth_value.empty() && id_group <= 0)
      throw userver::server::handlers::ClientError(HandlerErrorCode::kUnauthorized);

    if (!auth_value.empty())
    {
      const auto bearer_sep_pos = auth_value.find(' ');
      if (bearer_sep_pos == std::string::npos || std::string_view{auth_value.data(), bearer_sep_pos} != "Bearer")
        throw userver::server::handlers::ClientError(HandlerErrorCode::kUnauthorized);
      const auto token{auth_value.data() + bearer_sep_pos + 1};
      id_group = checkToken(token);
    }

    if (id_group <= 0)
      throw userver::server::handlers::ClientError(HandlerErrorCode::kUnauthorized);

    HashMap<std::string_view, std::function<void(int32_t, const userver::formats::json::Value&)>> no_content_methods = {
      {METHOD_ADD_STREAM, [this](auto&& id_group, auto&& json)
        {
          addStream(id_group, json);
        }},
      {METHOD_REMOVE_STREAM, [this](auto&& id_group, auto&& json)
        {
          removeStream(id_group, json);
        }},
      {METHOD_START_WORKFLOW, [this](auto&& id_group, auto&& json)
        {
          startWorkflow(id_group, json);
        }},
      {METHOD_STOP_WORKFLOW, [this](auto&& id_group, auto&& json)
        {
          stopWorkflow(id_group, json);
        }},
      {METHOD_SET_STREAM_DEFAULT_CONFIG, [this](auto&& id_group, auto&& json)
        {
          setStreamDefaultConfigParams(id_group, json);
        }},
    };

    if (no_content_methods.contains(api_method))
    {
      no_content_methods[api_method](id_group, json);
      request.SetResponseStatus(userver::server::http::HttpStatus::kNoContent);
      return {};
    }

    HashMap<std::string_view, std::function<userver::formats::json::Value(int32_t, const userver::formats::json::Value&)>> with_content_methods = {
      {METHOD_LIST_STREAMS, [this](auto&& id_group, auto&&)
        {
          return listStreams(id_group);
        }},
      {METHOD_GET_EVENT_DATA, [this](auto&& id_group, auto&& json)
        {
          return getEventData(id_group, json);
        }},
      {METHOD_GET_STREAM_DEFAULT_CONFIG, [this](auto&& id_group, auto&&)
        {
          return getStreamDefaultConfigParams(id_group);
        }},
    };

    if (with_content_methods.contains(api_method))
    {
      auto data = with_content_methods[api_method](id_group, json);
      if (data.IsEmpty())
      {
        request.SetResponseStatus(userver::server::http::HttpStatus::kNoContent);
        return {};
      }

      userver::formats::json::ValueBuilder response;
      response[PARAM_CODE] = std::to_string(userver::server::http::HttpStatus::kOk);
      response[PARAM_MESSAGE] = MESSAGE_OK;
      response[PARAM_DATA] = std::move(data);

      return response.ExtractValue();
    }

    throw userver::server::handlers::ClientError(HandlerErrorCode::kClientError,
      userver::server::handlers::ExternalBody{ERROR_NO_METHOD});
  }

  int32_t Api::checkToken(const absl::string_view token) const
  {
    if (const auto g_cache = groups_cache_.Get(); g_cache->contains(token))
      return g_cache->at(token).id_group;

    return -1;
  }

  void Api::requireMemberThrow(const userver::formats::json::Value& json, const absl::string_view member)
  {
    if (!json.HasMember(member))
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Required member `$0` not found.", member)});

    if (json[member].IsNull())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must not be null.", member)});

    if (json[member].IsArray())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must not be an array.", member)});

    if (json[member].IsObject())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must not be an object.", member)});

    if (json[member].IsString() && json[member].As<std::string>().empty())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must not be empty.", member)});
  }

  void Api::addStream(int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, PARAM_STREAM_ID);
    auto ext_id = json[PARAM_STREAM_ID].As<std::string>();
    if (json.HasMember(PARAM_CONFIG) && !json[PARAM_CONFIG].IsObject())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Invalid member `$0`.", PARAM_CONFIG)});

    userver::formats::json::Value config = {};
    if (json.HasMember(PARAM_CONFIG))
      config = json[PARAM_CONFIG];

    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      const userver::storages::postgres::Query query_check{SQL_GET_STREAM};
      if (auto res = trx.Execute(query_check, id_group, ext_id); res.IsEmpty())
      {
        const userver::storages::postgres::Query query{SQL_ADD_STREAM};
        trx.Execute(query, id_group, ext_id, config);
      } else
      {
        const userver::storages::postgres::Query query{SQL_UPDATE_STREAM};
        trx.Execute(query, config, res.AsSingleRow<int32_t>());
      }
      trx.Commit();
    } catch (...)
    {
      trx.Rollback();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  void Api::removeStream(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, PARAM_STREAM_ID);
    const auto ext_id = json[PARAM_STREAM_ID].As<std::string>();
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      const userver::storages::postgres::Query query{SQL_REMOVE_STREAM};
      trx.Execute(query, id_group, ext_id);
      trx.Commit();
    } catch (...)
    {
      trx.Rollback();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::listStreams(const int32_t id_group) const
  {
    userver::formats::json::ValueBuilder vstreams;
    try
    {
      const userver::storages::postgres::Query query{SQL_LIST_STREAMS};
      for (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, query, id_group); const auto& row : result)
      {
        userver::formats::json::ValueBuilder v;
        v[PARAM_STREAM_ID] = row[0].As<std::string>();
        v[PARAM_CONFIG] = row[1].As<userver::formats::json::Value>();
        vstreams.PushBack(std::move(v));
      }
    } catch (...)
    {
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return vstreams.ExtractValue();
  }

  void Api::startWorkflow(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, PARAM_STREAM_ID);
    auto vstream_key = absl::Substitute("$0_$1", id_group, json[PARAM_STREAM_ID].As<std::string>());
    workflow_.startWorkflow(std::move(vstream_key));
  }

  void Api::stopWorkflow(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, PARAM_STREAM_ID);
    auto vstream_key = absl::Substitute("$0_$1", id_group, json[PARAM_STREAM_ID].As<std::string>());
    workflow_.stopWorkflow(std::move(vstream_key), false);
  }

  userver::formats::json::Value Api::getEventData(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    try
    {
      if (json.HasMember(PARAM_EVENT_ID))
      {
        const userver::storages::postgres::Query query{SQL_GET_EVENT_BY_ID};
        const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
          query, json[PARAM_EVENT_ID].As<int64_t>());
        if (!result.IsEmpty())
        {
          // check if access permitted to this event
          const auto id_vstream = result[0][DatabaseFields::ID_VSTREAM].As<int32_t>();
          int32_t id_vg = -1;
          // scope for accessing cache
          {
            if (const auto vg_cache = vstream_group_cache_.Get(); vg_cache->contains(id_vstream))
              id_vg = vg_cache->at(id_vstream).id_group;
          }
          if (id_vg == id_group)
          {
            userver::formats::json::ValueBuilder event_data = result[0][DatabaseFields::INFO].As<userver::formats::json::Value>();
            event_data[PARAM_EVENT_DATE] = result[0][DatabaseFields::LOG_DATE].As<userver::storages::postgres::TimePointTz>();
            return event_data.ExtractValue();
          }
        }
      } else
      {
        if (!json.HasMember(PARAM_STREAM_ID) && !json.HasMember(PARAM_EVENT_DATE))
          throw userver::server::handlers::ClientError(
            ExternalBody{absl::Substitute("Member `$0` or both `$1` and `$2` must exist in the request.", PARAM_EVENT_ID, PARAM_STREAM_ID, PARAM_EVENT_DATE)});
        requireMemberThrow(json, PARAM_STREAM_ID);
        requireMemberThrow(json, PARAM_EVENT_DATE);

        int32_t id_vstream;
        std::chrono::milliseconds event_log_before{};
        std::chrono::milliseconds event_log_after{};
        // scope for accessing cache
        {
          const auto vstream_key = absl::Substitute("$0_$1", id_group, json[PARAM_STREAM_ID].As<std::string>());
          const auto cache = vstreams_config_cache_.Get();
          if (!cache->getData().contains(vstream_key))
            return {};

          id_vstream = cache->getData().at(vstream_key).id_vstream;
          event_log_before = cache->getData().at(vstream_key).event_log_before;
          event_log_after = cache->getData().at(vstream_key).event_log_after;
        }
        const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
          absl::Substitute(SQL_GET_NEAREST_EVENT,
            id_vstream,
            json[PARAM_EVENT_DATE].As<std::string>(),
            event_log_before.count(),
            event_log_after.count()));
        if (!result.IsEmpty())
        {
          userver::formats::json::ValueBuilder event_data = result[0][DatabaseFields::INFO].As<userver::formats::json::Value>();
          event_data[PARAM_EVENT_DATE] = result[0][DatabaseFields::LOG_DATE].As<userver::storages::postgres::TimePointTz>();
          return event_data.ExtractValue();
        }
      }
    } catch (userver::server::handlers::ClientError&)
    {
      throw;
    } catch (...)
    {
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return {};
  }

  void Api::setStreamDefaultConfigParams(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    if (!json.IsObject())
      throw userver::server::handlers::ClientError(ExternalBody{"Body is not a valid JSON object."});
    try
    {
      auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_SET_STREAM_DEFAULT_CONFIG_PARAMS, id_group, json);
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::getStreamDefaultConfigParams(const int32_t id_group) const
  {
    userver::formats::json::ValueBuilder data;
    try
    {
      if (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_GET_STREAM_DEFAULT_CONFIG_PARAMS, id_group); !result.IsEmpty())
        data = result[0][DatabaseFields::CONFIG].As<userver::formats::json::Value>();
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }
}  // namespace Lprs
