#include <filesystem>
#include <fstream>

#include <userver/components/component_context.hpp>
#include <userver/formats/serialize/common_containers.hpp>

#include "converters.hpp"
#include "frs_api.hpp"

namespace Frs
{
  Api::Api(const userver::components::ComponentConfig& config,
    const userver::components::ComponentContext& context)
    : HttpHandlerJsonBase(config, context),
      pg_cluster_(context.FindComponent<userver::components::Postgres>(Workflow::kDatabase).GetCluster()),
      groups_cache_(context.FindComponent<GroupsCache>()),
      config_cache_(context.FindComponent<ConfigCache>()),
      vstreams_config_cache_(context.FindComponent<VStreamsConfigCache>()),
      sg_config_cache_(context.FindComponent<SGConfigCache>()),
      workflow_(context.FindComponent<Workflow>())
  {
  }

  userver::formats::json::Value Api::HandleRequestJsonThrow(const userver::server::http::HttpRequest& request,
    const userver::formats::json::Value& json,
    userver::server::request::RequestContext&) const
  {
    const auto& auth_value = request.GetHeader("Authorization");
    const auto& api_method = request.GetPathArg(0);
    LOG_INFO_TO(workflow_.getLogger()) << "API call from: " << request.GetRemoteAddress() << ";  method: " << api_method
      << ";  body: " << absl::ClippedSubstr(ToStableString(json), 0, 300);
    if (api_method.starts_with(SG_METHOD_PREFIX))
    {
      // check special group authorization
      if (auth_value.empty())
        throw userver::server::handlers::ClientError(HandlerErrorCode::kUnauthorized);

      const auto bearer_sep_pos = auth_value.find(' ');
      if (bearer_sep_pos == std::string::npos || std::string_view{auth_value.data(), bearer_sep_pos} != "Bearer")
        throw userver::server::handlers::ClientError(HandlerErrorCode::kUnauthorized);
      const auto token{auth_value.data() + bearer_sep_pos + 1};
      const auto id_sgroup = checkSGToken(token);
      if (id_sgroup <= 0)
        throw userver::server::handlers::ClientError(HandlerErrorCode::kUnauthorized);

      HashMap<std::string_view, std::function<void(int32_t, const userver::formats::json::Value&)>> no_content_methods = {
        {METHOD_SG_DELETE_FACES, [this](auto&& id_sgroup, auto&& json)
          {
            sgDeleteFaces(id_sgroup, json);
          }},
        {METHOD_SG_UPDATE_GROUP, [this](auto&& id_sgroup, auto&& json)
          {
            sgUpdateGroup(id_sgroup, json);
          }},
      };

      if (no_content_methods.contains(api_method))
      {
        no_content_methods[api_method](id_sgroup, json);
        request.SetResponseStatus(userver::server::http::HttpStatus::kNoContent);
        return {};
      }

      HashMap<std::string_view, std::function<userver::formats::json::Value(int32_t, const userver::formats::json::Value&)>> with_content_methods = {
        {METHOD_SG_REGISTER_FACE, [this](auto&& id_sgroup, auto&& json)
          {
            return sgRegisterFace(id_sgroup, json);
          }},
        {METHOD_SG_LIST_FACES, [this](auto&& id_sgroup, auto&&)
          {
            return sgListFaces(id_sgroup);
          }},
        {METHOD_SG_RENEW_TOKEN, [this](auto&& id_sgroup, auto&&)
          {
            return sgRenewToken(id_sgroup);
          }},
        {METHOD_SG_SEARCH_FACES, [this](auto&& id_sgroup, auto&& json)
          {
            return sgSearchFaces(id_sgroup, json);
          }},
      };

      if (with_content_methods.contains(api_method))
      {
        auto data = with_content_methods[api_method](id_sgroup, json);
        if (data.IsEmpty())
        {
          request.SetResponseStatus(userver::server::http::HttpStatus::kNoContent);
          return {};
        }

        userver::formats::json::ValueBuilder response;
        response[P_CODE] = std::to_string(userver::server::http::HttpStatus::kOk);
        response[P_MESSAGE] = MESSAGE_REQUEST_COMPLETED;
        response[P_DATA] = std::move(data);

        return response.ExtractValue();
      }

      throw userver::server::handlers::ClientError(ExternalBody{ERROR_UNKNOWN_METHOD});
    }

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
      {METHOD_MOTION_DETECTION, [this](auto&& id_group, auto&& json)
        {
          motionDetection(id_group, json);
        }},
      {METHOD_DOOR_IS_OPEN, [this](auto&& id_group, auto&& json)
        {
          doorIsOpen(id_group, json);
        }},
      {METHOD_ADD_FACES, [this](auto&& id_group, auto&& json)
        {
          addFaces(id_group, json);
        }},
      {METHOD_REMOVE_FACES, [this](auto&& id_group, auto&& json)
        {
          removeFaces(id_group, json);
        }},
      {METHOD_REMOVE_STREAM, [this](auto&& id_group, auto&& json)
        {
          removeVStream(id_group, json);
        }},
      {METHOD_DELETE_FACES, [this](auto&& id_group, auto&& json)
        {
          deleteFaces(id_group, json);
        }},
      {METHOD_TEST_IMAGE, [this](auto&& id_group, auto&& json)
        {
          testImage(id_group, json);
        }},
      {METHOD_UPDATE_SPECIAL_GROUP, [this](auto&& id_group, auto&& json)
        {
          updateSpecialGroup(id_group, json);
        }},
      {METHOD_DELETE_SPECIAL_GROUP, [this](auto&& id_group, auto&& json)
        {
          deleteSpecialGroup(id_group, json);
        }},
      {METHOD_SAVE_DNN_STATS_DATA, [this](auto&&, auto&&)
        {
          saveDNNStatsData();
        }},
      {METHOD_SET_COMMON_CONFIG, [this](auto&& id_group, auto&& json)
        {
          setCommonConfigParams(id_group, json);
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
      {METHOD_BEST_QUALITY, [this](auto&& id_group, auto&& json)
        {
          return bestQuality(id_group, json);
        }},
      {METHOD_LIST_ALL_FACES, [this](auto&& id_group, auto&&)
        {
          return listAllFaces(id_group);
        }},
      {METHOD_GET_EVENTS, [this](auto&& id_group, auto&& json)
        {
          return getEvents(id_group, json);
        }},
      {METHOD_REGISTER_FACE, [this](auto&& id_group, auto&& json)
        {
          return registerFace(id_group, json);
        }},
      {METHOD_PROCESS_FRAME, [this](auto&& id_group, auto&& json)
        {
          return processFrame(id_group, json);
        }},
      {METHOD_ADD_SPECIAL_GROUP, [this](auto&& id_group, auto&& json)
        {
          return addSpecialGroup(id_group, json);
        }},
        {METHOD_LIST_SPECIAL_GROUPS, [this](auto&& id_group, auto&&)
        {
          return listSpecialGroups(id_group);
        }},
      {METHOD_GET_COMMON_CONFIG, [this](auto&& id_group, auto&&)
        {
          return getCommonConfigParams(id_group);
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
      response[P_CODE] = std::to_string(userver::server::http::HttpStatus::kOk);
      response[P_MESSAGE] = MESSAGE_REQUEST_COMPLETED;
      response[P_DATA] = std::move(data);

      return response.ExtractValue();
    }

    throw userver::server::handlers::ClientError(ExternalBody{ERROR_UNKNOWN_METHOD});
  }

  int32_t Api::checkToken(const absl::string_view token) const
  {
    if (const auto g_cache = groups_cache_.Get(); g_cache->contains(token))
      return g_cache->at(token).id_group;

    return -1;
  }

  int32_t Api::checkSGToken(const absl::string_view token) const
  {
    if (const auto sg_config_cache = sg_config_cache_.Get(); sg_config_cache->getData().contains(token))
      return sg_config_cache->getData().at(token).id_special_group;

    return -1;
  }

  int32_t Api::getVStreamId(const int32_t id_group, const absl::string_view vstream_ext) const
  {
    try
    {
      if (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_GET_VSTREAM_ID, id_group, vstream_ext); !result.IsEmpty())
        return result[0][DatabaseFields::ID_VSTREAM].As<int32_t>();
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

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

  void Api::requireArrayThrow(const userver::formats::json::Value& json, const absl::string_view member)
  {
    if (!json.HasMember(member))
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Required array member `$0` not found.", member)});

    if (json[member].IsNull())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must not be null.", member)});

    if (json[member].IsObject())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must not be an object.", member)});

    if (!json[member].IsArray())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` must be an array.", member)});

    if (json[member].IsEmpty())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Array member `$0` must not be empty.", member)});
  }

  void Api::addStream(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    const auto vstream_ext = convertToString(json[P_STREAM_ID]);
    auto url = json[P_URL].As<std::optional<std::string>>(std::nullopt);
    auto callback_url = json[P_CALLBACK_URL].As<std::optional<std::string>>(std::nullopt);

    std::vector<int32_t> faces;
    if (json.HasMember(P_FACE_IDS))
      try
      {
        faces = json[P_FACE_IDS].As<std::vector<int32_t>>({});
      } catch (const std::exception& e)
      {
        throw userver::server::handlers::ClientError(ExternalBody{e.what()});
      }

    std::optional<userver::formats::json::Value> config = std::nullopt;
    if (json.HasMember(P_PARAMS) && json[P_PARAMS].IsArray())  // for compatibility with old API
      try
      {
        HashSet<std::string> int_params = {
          ConfigParams::MAX_CAPTURE_ERROR_COUNT};

        HashSet<std::string> float_params = {
          ConfigParams::BLUR,
          ConfigParams::BLUR_MAX,
          ConfigParams::TOLERANCE,
          ConfigParams::TITLE_HEIGHT_RATIO,
          ConfigParams::FACE_CONFIDENCE_THRESHOLD,
          ConfigParams::FACE_ENLARGE_SCALE,
          ConfigParams::FACE_CLASS_CONFIDENCE_THRESHOLD,
          ConfigParams::MARGIN};

        HashSet<std::string> string_params = {
          ConfigParams::CONF_OSD_DT_FORMAT,
          ConfigParams::DNN_FD_INFERENCE_SERVER,
          ConfigParams::DNN_FC_INFERENCE_SERVER,
          ConfigParams::DNN_FR_INFERENCE_SERVER,
          ConfigParams::TITLE};

        HashSet<std::string> time_params = {
          ConfigParams::BEST_QUALITY_INTERVAL_AFTER,
          ConfigParams::BEST_QUALITY_INTERVAL_BEFORE,
          ConfigParams::CAPTURE_TIMEOUT,
          ConfigParams::DELAY_AFTER_ERROR,
          ConfigParams::DELAY_BETWEEN_FRAMES,
          ConfigParams::OPEN_DOOR_DURATION,
          ConfigParams::WORKFLOW_TIMEOUT,
          ConfigParams::UNKNOWN_DESCRIPTOR_TTL};

        HashSet<std::string> bool_params = {
          ConfigParams::FLAG_SPAWNED_DESCRIPTORS};

        // build video stream config
        userver::formats::json::ValueBuilder config_builder;
        for (const auto& param : json[P_PARAMS].As<userver::formats::json::Value>())
        {
          constexpr auto param_name = "paramName";
          constexpr auto param_value = "paramValue";
          const auto& p_name = param[param_name].As<std::string>();

          if (int_params.contains(p_name))
          {
            config_builder[p_name] = param[param_value].As<int32_t>();
            continue;
          }

          if (float_params.contains(p_name))
          {
            config_builder[p_name] = param[param_value].As<float>();
            continue;
          }

          if (string_params.contains(p_name))
          {
            config_builder[p_name] = param[param_value].As<std::string>();
            continue;
          }

          if (time_params.contains(p_name))
          {
            config_builder[p_name] = absl::Substitute("$0ms", static_cast<int32_t>(lround(param[param_value].As<float>() * 1000.0)));
            continue;
          }

          if (bool_params.contains(p_name))
          {
            config_builder[p_name] = param[param_value].As<bool>();
            continue;
          }

          if (p_name == ConfigParams::LOGS_LEVEL)
          {
            HashMap<int32_t, std::string> logs_level_map = {
              {0, "error"},
              {1, "info"},
              {2, "trace"}};
            auto p_value = param[param_value].As<int32_t>();
            config_builder[p_name] = logs_level_map.contains(p_value) ? logs_level_map.at(p_value) : "info";
          }
        }
        if (!config_builder.IsEmpty())
          config = config_builder.ExtractValue();
      } catch (const std::exception& e)
      {
        throw userver::server::handlers::ClientError(ExternalBody{e.what()});
      }
    if (json.HasMember(P_CONFIG) && json[P_CONFIG].IsObject())
      config = json[P_CONFIG];

    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      int32_t id_vstream;
      if (const auto res = trx.Execute(SQL_GET_STREAM, id_group, vstream_ext); res.IsEmpty())
      {
        const auto r = trx.Execute(SQL_ADD_STREAM, id_group, vstream_ext, url, callback_url, config);
        id_vstream = r.AsSingleRow<int32_t>();
      } else
      {
        id_vstream = res[0][DatabaseFields::ID_VSTREAM].As<int32_t>();
        if (!url)
          url = res[0][DatabaseFields::URL].As<std::optional<std::string>>();
        if (!callback_url)
          callback_url = res[0][DatabaseFields::CALLBACK_URL].As<std::optional<std::string>>();
        trx.Execute(SQL_UPDATE_STREAM, id_group, url, callback_url, config, id_vstream);
      }

      if (!faces.empty())
      {
        // bind faces to video stream
        for (const auto& id_descriptor : faces)
          trx.Execute(SQL_ADD_LINK_DESCRIPTOR_VSTREAM, id_group, id_vstream, id_descriptor);
      }

      trx.Commit();
    } catch (const std::exception& e)
    {
      trx.Rollback();
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::listStreams(const int32_t id_group) const
  {
    struct VStreamData
    {
      std::optional<std::string> url;
      std::optional<std::string> callback_url;
      std::optional<userver::formats::json::Value> config;
      std::vector<int32_t> faces;
    };
    HashMap<std::string, VStreamData> vstreams_data;
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      for (const auto result = trx.Execute(SQL_LIST_STREAMS_SIMPLE, id_group); const auto& row : result)
      {
        auto vstream_ext = row[DatabaseFields::VSTREAM_EXT].As<std::string>();
        vstreams_data[vstream_ext] = {
          row[DatabaseFields::URL].As<std::optional<std::string>>(),
          row[DatabaseFields::CALLBACK_URL].As<std::optional<std::string>>(),
          row[DatabaseFields::CONFIG].As<std::optional<userver::formats::json::Value>>(),
          {}};
      }
      for (const auto result = trx.Execute(SQL_LIST_STREAM_FACES, id_group); const auto& row : result)
      {
        auto vstream_ext = row[DatabaseFields::VSTREAM_EXT].As<std::string>();
        auto id_descriptor = row[DatabaseFields::ID_DESCRIPTOR].As<int32_t>();
        if (vstreams_data.contains(vstream_ext))
          vstreams_data[vstream_ext].faces.push_back(id_descriptor);
      }
      trx.Commit();
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    userver::formats::json::ValueBuilder data;
    for (const auto& [fst, snd] : vstreams_data)
    {
      userver::formats::json::ValueBuilder v;
      v[P_STREAM_ID] = fst;
      if (snd.url)
        v[P_URL] = snd.url.value();
      if (snd.callback_url)
        v[P_CALLBACK_URL] = snd.callback_url.value();
      if (snd.config)
        v[P_CONFIG] = snd.config.value();
      if (!snd.faces.empty())
        v[P_FACE_IDS] = snd.faces;
      data.PushBack(std::move(v));
    }

    return data.ExtractValue();
  }

  void Api::motionDetection(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    requireMemberThrow(json, P_START);
    auto vstream_key = absl::Substitute("$0_$1", id_group, convertToString(json[P_STREAM_ID]));
    if (json[P_START].IsBool() ? json[P_START].As<bool>() : json[P_START].As<std::string>() == "t")
      workflow_.startWorkflow(std::move(vstream_key));
    else
      workflow_.stopWorkflow(std::move(vstream_key), false);
  }

  void Api::doorIsOpen(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    auto vstream_key = absl::Substitute("$0_$1", id_group, convertToString(json[P_STREAM_ID]));
    workflow_.stopWorkflow(std::move(vstream_key), false);
  }

  userver::formats::json::Value Api::bestQuality(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    const auto id_log = convertToInt<int32_t>(json[P_LOG_EVENT_ID]);
    if (!(id_log || (json.HasMember(P_STREAM_ID) && !json[P_STREAM_ID].IsNull() && json.HasMember(P_DATE) && !json[P_DATE].IsNull())))
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Required members `$0` or `$1` and `$2` not found or invalid.",
        P_LOG_EVENT_ID, P_STREAM_ID, P_DATE)});

    std::string vstream_key;
    if (json.HasMember(P_STREAM_ID) && !json[P_STREAM_ID].IsNull())
      vstream_key = absl::Substitute("$0_$1", id_group, convertToString(json[P_STREAM_ID]));
    const auto ext_event_uuid = json[P_EVENT_UUID].As<std::string>("");

    int32_t id_vstream{};
    std::chrono::milliseconds interval_before{};
    std::chrono::milliseconds interval_after{};
    bool do_copy_event_data = false;

    // scope for accessing cache
    {
      if (const auto cache = config_cache_.Get(); cache->getCommonConfig().contains(id_group))
        do_copy_event_data = cache->getCommonConfig().at(id_group).flag_copy_event_data;
    }

    if (!id_log)
    {
      const auto cache = vstreams_config_cache_.Get();
      if (!cache->getData().contains(vstream_key))
        return {};

      id_vstream = getVStreamId(id_group, convertToString(json[P_STREAM_ID]));
      interval_before = cache->getData().at(vstream_key).best_quality_interval_before;
      interval_after = cache->getData().at(vstream_key).best_quality_interval_after;
    }

    try
    {
      const auto result = (id_log)
                            ? pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_GET_LOG_FACE_BY_ID, id_group, id_log.value())
                            : pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
                                absl::Substitute(SQL_GET_LOG_FACE_BEST_QUALITY,
                                  id_vstream,
                                  json[P_DATE].As<std::string>(),
                                  interval_before.count(),
                                  interval_after.count()));
      if (!result.IsEmpty())
      {
        userver::formats::json::ValueBuilder event_data;
        event_data[Api::P_SCREENSHOT_URL] = result[0][DatabaseFields::SCREENSHOT_URL].As<std::string>();
        event_data[Api::P_FACE_LEFT] = result[0][DatabaseFields::FACE_LEFT].As<int32_t>();
        event_data[Api::P_FACE_TOP] = result[0][DatabaseFields::FACE_TOP].As<int32_t>();
        event_data[Api::P_FACE_WIDTH] = result[0][DatabaseFields::FACE_WIDTH].As<int32_t>();
        event_data[Api::P_FACE_HEIGHT] = result[0][DatabaseFields::FACE_HEIGHT].As<int32_t>();

        const auto id_event_log = result[0][DatabaseFields::ID_LOG].As<int32_t>();

        // schedule copy event data
        if (const auto copy_event_data = result[0][DatabaseFields::COPY_EVENT_DATA].As<int16_t>(); do_copy_event_data && (copy_event_data == CopyEventData::NONE) && !ext_event_uuid.empty())
          pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_SET_COPY_DATA_BY_ID,
            static_cast<int16_t>(CopyEventData::SCHEDULED), ext_event_uuid, id_event_log);

        return event_data.ExtractValue();
      }
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return {};
  }

  void Api::addFaces(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    requireArrayThrow(json, P_FACE_IDS);

    std::vector<int32_t> faces;
    if (json.HasMember(P_FACE_IDS))
      try
      {
        faces = json[P_FACE_IDS].As<std::vector<int32_t>>({});
      } catch (const std::exception& e)
      {
        throw userver::server::handlers::ClientError(ExternalBody{e.what()});
      }

    if (const auto id_vstream = getVStreamId(id_group, convertToString(json[P_STREAM_ID])); !faces.empty() && id_vstream > 0)
    {
      auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
      try
      {
        // bind faces to video stream
        for (const auto& id_descriptor : faces)
          trx.Execute(SQL_ADD_LINK_DESCRIPTOR_VSTREAM, id_group, id_vstream, id_descriptor);
        trx.Commit();
      } catch (const std::exception& e)
      {
        trx.Rollback();
        LOG_ERROR_TO(workflow_.getLogger()) << e.what();
        throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
      }
    }
  }

  void Api::removeFaces(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    requireArrayThrow(json, P_FACE_IDS);

    std::vector<int32_t> faces;
    if (json.HasMember(P_FACE_IDS))
      try
      {
        faces = json[P_FACE_IDS].As<std::vector<int32_t>>({});
      } catch (const std::exception& e)
      {
        throw userver::server::handlers::ClientError(ExternalBody{e.what()});
      }

    if (const auto id_vstream = getVStreamId(id_group, convertToString(json[P_STREAM_ID])); !faces.empty() && id_vstream > 0)
    {
      auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
      try
      {
        // unbind faces from video stream: we don't delete rows from database right now, just mark them to delete later
        for (const auto& id_descriptor : faces)
          trx.Execute(SQL_REMOVE_LINK_DESCRIPTOR_VSTREAM, id_vstream, id_descriptor);
        trx.Commit();
      } catch (const std::exception& e)
      {
        trx.Rollback();
        LOG_ERROR_TO(workflow_.getLogger()) << e.what();
        throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
      }
    }
  }

  void Api::removeVStream(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);

    const auto id_vstream = getVStreamId(id_group, convertToString(json[P_STREAM_ID]));
    if (id_vstream <= 0)
      return;

    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      trx.Execute(SQL_REMOVE_LINK_DESCRIPTOR_VSTREAM_BY_VSTREAM, id_vstream);
      trx.Execute(SQL_DELETE_VIDEO_STREAM, id_group, id_vstream);
      trx.Commit();
    } catch (const std::exception& e)
    {
      trx.Rollback();
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::listAllFaces(const int32_t id_group) const
  {
    userver::formats::json::ValueBuilder data;
    try
    {
      for (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_LIST_ALL_FACES, id_group); const auto& row : result)
        data.PushBack(row[DatabaseFields::ID_DESCRIPTOR].As<int32_t>());
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }

  void Api::deleteFaces(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireArrayThrow(json, P_FACE_IDS);

    std::vector<int32_t> faces;
    if (json.HasMember(P_FACE_IDS))
      try
      {
        faces = json[P_FACE_IDS].As<std::vector<int32_t>>({});
      } catch (const std::exception& e)
      {
        throw userver::server::handlers::ClientError(ExternalBody{e.what()});
      }

    if (!faces.empty())
    {
      auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
      try
      {
        for (const auto& id_descriptor : faces)
        {
          // unbind faces from all video streams: we don't delete rows from database right now, just mark them to delete later
          trx.Execute(SQL_REMOVE_LINK_DESCRIPTOR_VSTREAM_BY_DESCRIPTOR, id_group, id_descriptor);

          // do not delete descriptor from database, just mark for deletion
          trx.Execute(SQL_REMOVE_DESCRIPTOR, id_group, id_descriptor);

          // do not delete spawned descriptors from database, just mark for deletion
          trx.Execute(SQL_REMOVE_SPAWNED_DESCRIPTORS, id_group, id_descriptor);
        }
        trx.Commit();
      } catch (const std::exception& e)
      {
        trx.Rollback();
        LOG_ERROR_TO(workflow_.getLogger()) << e.what();
        throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
      }
    }
  }

  userver::formats::json::Value Api::getEvents(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    requireMemberThrow(json, P_DATE_START);
    requireMemberThrow(json, P_DATE_END);

    const auto id_vstream = getVStreamId(id_group, convertToString(json[P_STREAM_ID]));
    userver::formats::json::ValueBuilder data;
    try
    {
      const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
        absl::Substitute(SQL_GET_LOG_FACES_FROM_INTERVAL,
          id_vstream,
          json[P_DATE_START].As<std::string>(),
          json[P_DATE_END].As<std::string>()));
      for (const auto& row : result)
      {
        userver::formats::json::ValueBuilder v;
        v[P_DATE] = row[DatabaseFields::LOG_DATE].As<userver::storages::postgres::TimePointTz>();
        if (!row[DatabaseFields::ID_DESCRIPTOR].IsNull())
          v[P_FACE_ID] = row[DatabaseFields::ID_DESCRIPTOR].As<int32_t>();
        v[P_QUALITY] = row[DatabaseFields::QUALITY].As<double>();
        v[P_SCREENSHOT_URL] = row[DatabaseFields::SCREENSHOT_URL].As<std::string>();
        v[P_FACE_LEFT] = row[DatabaseFields::FACE_LEFT].As<int32_t>();
        v[P_FACE_TOP] = row[DatabaseFields::FACE_TOP].As<int32_t>();
        v[P_FACE_WIDTH] = row[DatabaseFields::FACE_WIDTH].As<int32_t>();
        v[P_FACE_HEIGHT] = row[DatabaseFields::FACE_HEIGHT].As<int32_t>();
        data.PushBack(std::move(v));
      }
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }

  userver::formats::json::Value Api::registerFace(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    requireMemberThrow(json, P_URL);

    auto vstream_key = absl::Substitute("$0_$1", id_group, convertToString(json[P_STREAM_ID]));
    TaskData task_data{
      .id_group = id_group,
      .vstream_key = std::move(vstream_key),
      .task_type = TaskType::TASK_REGISTER_DESCRIPTOR,
      .frame_url = json[P_URL].As<std::string>()};
    task_data.face_left = json[P_FACE_LEFT].As<int>(0);
    task_data.face_top = json[P_FACE_TOP].As<int>(0);
    task_data.face_width = json[P_FACE_WIDTH].As<int>(0);
    task_data.face_height = json[P_FACE_HEIGHT].As<int>(0);
    auto [id_descriptor, comments, face_image, face_left, face_top, face_width, face_height, id_descriptors] = workflow_.processPipeline(std::move(task_data));
    userver::formats::json::ValueBuilder data;
    if (id_descriptor > 0)
    {
      std::vector<uchar> buff;
      imencode(".jpg", face_image, buff);
      data[P_FACE_ID] = id_descriptor;
      data[P_FACE_LEFT] = face_left;
      data[P_FACE_TOP] = face_top;
      data[P_FACE_WIDTH] = face_width;
      data[P_FACE_HEIGHT] = face_height;
      data[P_FACE_IMAGE] = absl::Substitute("data:$0;base64,$1", Workflow::MIME_IMAGE,
        absl::Base64Escape(std::string(reinterpret_cast<const char*>(buff.data()), buff.size())));
    } else
      throw userver::server::handlers::ClientError(ExternalBody{comments});

    return data.ExtractValue();
  }

  void Api::testImage(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_STREAM_ID);
    requireMemberThrow(json, P_URL);

    auto vstream_key = absl::Substitute("$0_$1", id_group, convertToString(json[P_STREAM_ID]));
    TaskData task_data{
      .id_group = id_group,
      .vstream_key = std::move(vstream_key),
      .task_type = TaskType::TASK_TEST,
      .frame_url = json[P_URL].As<std::string>()};
    workflow_.processPipeline(std::move(task_data));
  }

  userver::formats::json::Value Api::processFrame(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    if (!((json.HasMember(P_STREAM_ID) && !json[P_STREAM_ID].IsNull()) || (json.HasMember(P_SG_ID) && !json[P_SG_ID].IsNull() && json[P_SG_ID].IsInt())))
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Required members `$0` or `$1` not found or invalid.",
        P_STREAM_ID, P_SG_ID)});
    requireMemberThrow(json, P_URL);

    std::string vstream_key;
    if (json.HasMember(P_STREAM_ID))
      vstream_key = absl::Substitute("$0_$1", id_group, convertToString(json[P_STREAM_ID]));
    TaskData task_data{
      .id_group = id_group,
      .vstream_key = std::move(vstream_key),
      .task_type = TaskType::TASK_PROCESS_FRAME,
      .frame_url = json[P_URL].As<std::string>()};

    if (json.HasMember(P_SG_ID) && vstream_key.empty())
      task_data.id_sgroup = json[P_SG_ID].As<int32_t>();
    auto [id_descriptor, comments, face_image, face_left, face_top, face_width, face_height, id_descriptors] = workflow_.processPipeline(std::move(task_data));
    userver::formats::json::ValueBuilder data = id_descriptors;

    return data.ExtractValue();
  }

  userver::formats::json::Value Api::addSpecialGroup(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_SPECIAL_GROUP_NAME);

    const auto group_name = json[P_SPECIAL_GROUP_NAME].As<std::string>();

    int32_t max_descriptor_count{1};
    // scope for accessing cache
    {
      if (const auto cache = config_cache_.Get(); cache->getCommonConfig().contains(id_group))
        max_descriptor_count = cache->getCommonConfig().at(id_group).sg_max_descriptor_count;
    }

    userver::formats::json::ValueBuilder data;
    if (json.HasMember(P_MAX_DESCRIPTOR_COUNT))
    {
      if (!json[P_MAX_DESCRIPTOR_COUNT].IsInt())
        throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` is invalid.",
          P_MAX_DESCRIPTOR_COUNT)});
      max_descriptor_count = std::max(1, std::min(json[P_MAX_DESCRIPTOR_COUNT].As<int32_t>(), max_descriptor_count));
    }

    try
    {
      const auto res = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
        SQL_ADD_SPECIAL_GROUP, id_group, group_name, max_descriptor_count);
      data[P_SG_ID] = res[0][DatabaseFields::ID_SPECIAL_GROUP].As<int32_t>();
      data[P_SG_API_TOKEN] = res[0][DatabaseFields::SG_API_TOKEN].As<std::string>();
    } catch (userver::storages::postgres::UniqueViolation& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(ExternalBody{"A special group with this name already exists."});
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }

  void Api::updateSpecialGroup(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_SG_ID);
    if (!json[P_SG_ID].IsInt())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` is invalid.", P_SG_ID)});

    if (json.HasMember(P_SPECIAL_GROUP_NAME) && !json[P_SPECIAL_GROUP_NAME].IsString())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` is invalid.", P_SPECIAL_GROUP_NAME)});

    if (json.HasMember(P_MAX_DESCRIPTOR_COUNT) && !json[P_MAX_DESCRIPTOR_COUNT].IsInt())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` is invalid.", P_MAX_DESCRIPTOR_COUNT)});

    const auto id_sgroup = json[P_SG_ID].As<int32_t>();
    std::string group_name = DatabaseFields::SG_NAME;
    if (json.HasMember(P_SPECIAL_GROUP_NAME))
      group_name = absl::Substitute("'$0'", absl::StrReplaceAll(json[P_SPECIAL_GROUP_NAME].As<std::string>(), {{"'", "''"}}));
    std::string max_descriptor_count = DatabaseFields::SG_MAX_DESCRIPTOR_COUNT;
    if (json.HasMember(P_MAX_DESCRIPTOR_COUNT))
      max_descriptor_count = std::to_string(json[P_MAX_DESCRIPTOR_COUNT].As<int32_t>());

    try
    {
      pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
        absl::Substitute(SQL_UPDATE_SPECIAL_GROUP, group_name, max_descriptor_count, id_group, id_sgroup));
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  void Api::deleteSpecialGroup(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_SG_ID);
    if (!json[P_SG_ID].IsInt())
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Member `$0` is invalid.", P_SG_ID)});

    const auto id_sgroup = json[P_SG_ID].As<int32_t>();
    auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
    try
    {
      // do not delete data from database, just mark for deletion
      trx.Execute(SQL_REMOVE_LINK_DESCRIPTOR_SG_ALL, id_sgroup);
      trx.Execute(SQL_REMOVE_SG_FACE_DESCRIPTORS, id_sgroup);
      trx.Execute(SQL_DELETE_SPECIAL_GROUP, id_group, id_sgroup);
      trx.Commit();
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::listSpecialGroups(const int32_t id_group) const
  {
    userver::formats::json::ValueBuilder data;
    try
    {
      for (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_LIST_SPECIAL_GROUPS, id_group); const auto& row : result)
      {
        userver::formats::json::ValueBuilder v;
        v[P_SG_ID] = row[DatabaseFields::ID_SPECIAL_GROUP].As<int32_t>();
        v[P_SPECIAL_GROUP_NAME] = row[DatabaseFields::SG_NAME].As<std::string>();
        v[P_SG_API_TOKEN] = row[DatabaseFields::SG_API_TOKEN].As<std::string>();
        if (!row[DatabaseFields::CALLBACK_URL].IsNull())
          v[P_CALLBACK_URL] = row[DatabaseFields::CALLBACK_URL].As<std::string>();
        v[P_MAX_DESCRIPTOR_COUNT] = row[DatabaseFields::SG_MAX_DESCRIPTOR_COUNT].As<int32_t>();
        data.PushBack(std::move(v));
      }
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }


  void Api::saveDNNStatsData() const
  {
    workflow_.saveDNNStatsData();
  }

  void Api::setCommonConfigParams(const int32_t id_group, const userver::formats::json::Value& json) const
  {
    if (!json.IsObject())
      throw userver::server::handlers::ClientError(ExternalBody{"Body is not a valid JSON object."});
    try
    {
      auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_SET_COMMON_CONFIG_PARAMS, id_group, json);
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::getCommonConfigParams(const int32_t id_group) const
  {
    userver::formats::json::ValueBuilder data;
    try
    {
      if (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_GET_COMMON_CONFIG_PARAMS, id_group); !result.IsEmpty())
        data = result[0][DatabaseFields::CONFIG].As<userver::formats::json::Value>();
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
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

  // member functions for special group methods
  userver::formats::json::Value Api::sgRegisterFace(const int32_t id_sgroup, const userver::formats::json::Value& json) const
  {
    requireMemberThrow(json, P_URL);

    int32_t id_group = -1;
    // scope for accessing cache
    {
      if (const auto sg_config = sg_config_cache_.Get(); sg_config->getMap().contains(id_sgroup))
        id_group = sg_config->getData().at(sg_config->getMap().at(id_sgroup)).id_group;
    }
    if (id_group <= 0)
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);

    TaskData task_data{
      .id_group = id_group,
      .vstream_key = {},
      .task_type = TaskType::TASK_REGISTER_DESCRIPTOR,
      .frame_url = json[P_URL].As<std::string>(),
      .id_sgroup = id_sgroup};
    task_data.face_left = json[P_FACE_LEFT].As<int>(0);
    task_data.face_top = json[P_FACE_TOP].As<int>(0);
    task_data.face_width = json[P_FACE_WIDTH].As<int>(0);
    task_data.face_height = json[P_FACE_HEIGHT].As<int>(0);
    auto [id_descriptor, comments, face_image, face_left, face_top, face_width, face_height, id_descriptors] = workflow_.processPipeline(std::move(task_data));
    userver::formats::json::ValueBuilder data;
    if (id_descriptor > 0)
    {
      std::vector<uchar> buff;
      imencode(".jpg", face_image, buff);
      data[P_FACE_ID] = id_descriptor;
      data[P_FACE_LEFT] = face_left;
      data[P_FACE_TOP] = face_top;
      data[P_FACE_WIDTH] = face_width;
      data[P_FACE_HEIGHT] = face_height;
      data[P_FACE_IMAGE] = absl::Substitute("data:$0;base64,$1", Workflow::MIME_IMAGE,
        absl::Base64Escape(std::string(reinterpret_cast<const char*>(buff.data()), buff.size())));
    } else
      throw userver::server::handlers::ClientError(ExternalBody{comments});

    return data.ExtractValue();
  }

  void Api::sgDeleteFaces(const int32_t id_sgroup, const userver::formats::json::Value& json) const
  {
    requireArrayThrow(json, P_FACE_IDS);

    std::vector<int32_t> faces;
    if (json.HasMember(P_FACE_IDS))
      try
      {
        faces = json[P_FACE_IDS].As<std::vector<int32_t>>({});
      } catch (const std::exception& e)
      {
        throw userver::server::handlers::ClientError(ExternalBody{e.what()});
      }

    if (!faces.empty())
    {
      auto trx = pg_cluster_->Begin(userver::storages::postgres::ClusterHostType::kMaster, {});
      try
      {
        for (const auto& id_descriptor : faces)
        {
          // unbind face from special group: we don't delete row from database right now, just mark for deletion
          trx.Execute(SQL_REMOVE_LINK_DESCRIPTOR_SG, id_sgroup, id_descriptor);

          // do not delete descriptor from database, just mark for deletion
          trx.Execute(SQL_REMOVE_SG_FACE_DESCRIPTOR, id_sgroup, id_descriptor);
        }
        trx.Commit();
      } catch (const std::exception& e)
      {
        trx.Rollback();
        LOG_ERROR_TO(workflow_.getLogger()) << e.what();
        throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
      }
    }
  }

  userver::formats::json::Value Api::sgListFaces(const int32_t id_sgroup) const
  {
    userver::formats::json::ValueBuilder data;
    try
    {
      for (const auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_SG_LIST_FACES, id_sgroup); const auto& row : result)
      {
        userver::formats::json::ValueBuilder item;
        item[P_FACE_ID] = row[DatabaseFields::ID_DESCRIPTOR].As<int32_t>();
        item[P_FACE_IMAGE] = row[DatabaseFields::FACE_IMAGE].As<std::string>();
        data.PushBack(std::move(item));
      }
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }

  void Api::sgUpdateGroup(const int32_t id_sgroup, const userver::formats::json::Value& json) const
  {
    requireArrayThrow(json, P_CALLBACK_URL);

    const auto callback_url = json[P_CALLBACK_URL].As<std::string>();
    try
    {
      pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_SG_UPDATE_GROUP, id_sgroup, callback_url);
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }
  }

  userver::formats::json::Value Api::sgRenewToken(const int32_t id_sgroup) const
  {
    userver::formats::json::ValueBuilder data;
    try
    {
      const auto r = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster, SQL_SG_RENEW_TOKEN, id_sgroup);
      data[P_SG_API_TOKEN] = r.AsSingleRow<std::string>();
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    return data.ExtractValue();
  }

  userver::formats::json::Value Api::sgSearchFaces(int32_t id_sgroup, const userver::formats::json::Value& json) const
  {
    int32_t id_group = -1;
    // scope for accessing cache
    {
      if (auto sg_config = sg_config_cache_.Get(); sg_config->getMap().contains(id_sgroup))
        id_group = sg_config->getData().at(sg_config->getMap().at(id_sgroup)).id_group;
    }
    if (id_group <= 0)
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);

    requireArrayThrow(json, P_FACE_IDS);
    requireMemberThrow(json, P_DATE_START);
    requireMemberThrow(json, P_DATE_END);
    requireMemberThrow(json, P_SIMILARITY_THRESHOLD);

    auto flag_logs = true;
    if (json.HasMember(P_SEARCH_IN_LOGS))
      flag_logs = json[P_SEARCH_IN_LOGS].As<bool>(flag_logs);
    auto flag_events = true;
    if (json.HasMember(P_SEARCH_IN_EVENTS))
      flag_events = json[P_SEARCH_IN_EVENTS].As<bool>(flag_events);
    if (!flag_logs && !flag_events)
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute(
        "At least one of the members `$0` or `$1` must be true.", P_SEARCH_IN_LOGS, P_SEARCH_IN_EVENTS)});

    std::string err;
    absl::Time date_start;
    if (!absl::ParseTime(Workflow::DATE_FORMAT, json[P_DATE_START].As<std::string>(), &date_start, &err))
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Required member `$0` is invalid.", P_DATE_START)});
    absl::Time date_end;
    if (!absl::ParseTime(Workflow::DATE_FORMAT, json[P_DATE_END].As<std::string>(), &date_end, &err))
      throw userver::server::handlers::ClientError(ExternalBody{absl::Substitute("Required member `$0` is invalid.", P_DATE_END)});
    date_end += absl::Hours(24);
    std::vector<int32_t> faces;
    float similarity_threshold = 0.5f;
    try
    {
      faces = json[P_FACE_IDS].As<std::vector<int32_t>>({});
      similarity_threshold = json[P_SIMILARITY_THRESHOLD].As<float>(similarity_threshold);
    } catch (const std::exception& e)
    {
      throw userver::server::handlers::ClientError(ExternalBody{e.what()});
    }

    HashMap<int32_t, DescriptorData> descriptors;
    try
    {
      auto result = pg_cluster_->Execute(userver::storages::postgres::ClusterHostType::kMaster,
        absl::Substitute(SQL_SG_DESCRIPTORS, id_sgroup, absl::StrJoin(faces, ",")));
      for (const auto& row : result)
      {
        auto id_descriptor = row[DatabaseFields::ID_DESCRIPTOR].As<int32_t>();
        std::string descriptor_data;
        row[DatabaseFields::DESCRIPTOR_DATA].To(userver::storages::postgres::Bytea(descriptor_data));
        descriptors[id_descriptor] = {};
        std::memmove(descriptors[id_descriptor].data, descriptor_data.data(), descriptor_data.size());
      }
    } catch (const std::exception& e)
    {
      LOG_ERROR_TO(workflow_.getLogger()) << e.what();
      throw userver::server::handlers::ClientError(HandlerErrorCode::kServerSideError);
    }

    std::vector<ResultItem> search_results;
    HashSet<std::string> event_ids;

    if (const auto search_path = absl::Substitute("$0group_$1/", workflow_.getLocalConfig().events_path, id_group); flag_events && std::filesystem::exists(search_path))
    {
      auto search_start_date = absl::FormatTime(Workflow::DATE_FORMAT, date_start, absl::LocalTimeZone());
      auto search_end_date = absl::FormatTime(Workflow::DATE_FORMAT, date_end, absl::LocalTimeZone());
      for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(search_path))
        if (dir_entry.is_regular_file() && dir_entry.path().extension().string() == Workflow::DATA_FILE_SUFFIX
            && dir_entry.path().filename() >= absl::StrCat(search_start_date, Workflow::DATA_FILE_SUFFIX)
            && dir_entry.path().filename() <= absl::StrCat(search_end_date, Workflow::DATA_FILE_SUFFIX))
        {
          // for test
          // cout << dir_entry.path().filename().string() << "\n";

          std::error_code ec;
          const auto f_size = std::filesystem::file_size(dir_entry.path(), ec);
          EventData data{};
          if (!ec && f_size > 0)
          {
            std::string s_data(f_size, '\0');
            std::ifstream fr_data(dir_entry.path(), std::ios::in | std::ios::binary);
            while (fr_data.good())
            {
              fr_data.read(reinterpret_cast<char*>(&data), sizeof(data));
              if (fr_data.gcount() == sizeof(data))
                for (auto& [fst, snd] : descriptors)
                {
                  if (double cosine_distance = cosineDistanceSIMD(snd.data, data.data); cosine_distance > similarity_threshold)
                  {
                    auto event_id = std::string(data.event_id, sizeof(data.event_id));
                    event_ids.insert(event_id);

                    // open json file with event identifier
                    std::string json_filename = absl::Substitute("$0group_$1/$2/$3/$4/$5/$6.json", workflow_.getLocalConfig().events_path,
                      id_group, event_id[0], event_id[1], event_id[2], event_id[3], event_id);
                    if (!std::filesystem::exists(json_filename, ec))
                    {
                      // trying the shorter path for compatibility with the old project
                      json_filename = absl::Substitute("$0group_$1/$2/$3/$4/$5.json", workflow_.getLocalConfig().events_path,
                      id_group, event_id[0], event_id[1], event_id[2], event_id);
                    }
                    if (auto json_size = std::filesystem::file_size(json_filename, ec); !ec && json_size > 0)
                    {
                      std::ifstream f_json(json_filename, std::ios::binary);
                      auto event_json = userver::formats::json::FromStream(f_json);
                      auto uuid = event_json["event_uuid"].As<std::string>("");
                      std::string image_url;
                      if (uuid.empty())
                      {
                        image_url = absl::Substitute("$0group_$1/$2/$3/$4/$5/$6.jpg", workflow_.getLocalConfig().screenshots_url_prefix,
                          id_group, event_id[0], event_id[1], event_id[2], event_id[3], event_id);
                      }
                      auto event_date = event_json["event_date"].As<std::string>(dir_entry.path().filename().stem().string());
                      search_results.push_back({event_date,
                        event_id,
                        uuid,
                        image_url,
                        fst,
                        cosine_distance});
                    }
                  }
                }
            }
          }
        }
    }

    if (const auto search_path = absl::Substitute("$0group_$1/", workflow_.getLocalConfig().screenshots_path, id_group); flag_logs && std::filesystem::exists(search_path))
      for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(search_path))
        if (dir_entry.is_regular_file() && dir_entry.path().extension().string() == Workflow::DATA_FILE_SUFFIX
            && std::chrono::file_clock::to_sys(dir_entry.last_write_time()) >= absl::ToChronoTime(date_start)
            && std::chrono::file_clock::to_sys(dir_entry.last_write_time()) < absl::ToChronoTime(date_end))
        {
          std::error_code ec;
          const auto f_size = std::filesystem::file_size(dir_entry.path(), ec);
          EventData data{};
          if (!ec && f_size > 0)
          {
            std::string s_data(f_size, '\0');
            std::ifstream fr_data(dir_entry.path(), std::ios::in | std::ios::binary);
            while (fr_data.good())
            {
              fr_data.read(reinterpret_cast<char*>(&data), sizeof(data));
              if (fr_data.gcount() == sizeof(data))
                for (auto& [fst, snd] : descriptors)
                {
                  if (double cosine_distance = cosineDistanceSIMD(snd.data, data.data); cosine_distance > similarity_threshold)
                  {
                    auto event_id = std::string(data.event_id, sizeof(data.event_id));

                    // if a log entry is included in the events, then we ignore it
                    if (event_ids.find(event_id) != event_ids.end())
                      continue;

                    // open json log file
                    std::string f_name = dir_entry.path().stem();
                    std::string json_filename = dir_entry.path().parent_path() / (f_name + std::string(Workflow::JSON_SUFFIX));
                    if (auto json_size = std::filesystem::file_size(json_filename, ec); !ec && json_size > 0)
                    {
                      std::ifstream f_json(json_filename, std::ios::binary);
                      if (auto event_json = userver::formats::json::FromStream(f_json); event_json.HasMember("event_date"))
                      {
                        auto event_date = event_json["event_date"].As<std::string>();
                        std::string image_url = absl::Substitute("$0group_$1/$2/$3/$4/$5/$6.jpg", workflow_.getLocalConfig().screenshots_url_prefix,
                          id_group, f_name[0], f_name[1], f_name[2], f_name[3], f_name);
                        search_results.push_back({event_date,
                          event_id,
                          "",
                          image_url,
                          fst,
                          cosine_distance});
                      }
                    }
                  }
                }
            }
          }
        }

    std::ranges::sort(search_results, std::greater());
    userver::formats::json::ValueBuilder json_data;
    for (const auto& [event_date, event_id, uuid, url_image, id_descriptor, similarity] : search_results)
    {
      userver::formats::json::ValueBuilder v;
      v[P_DATE] = event_date;
      v[P_EVENT_UUID] = uuid;
      v[P_LOG_EVENT_ID] = event_id;
      v[P_URL] = url_image;
      v[P_FACE_ID] = id_descriptor;
      v[P_SIMILARITY] = similarity;
      json_data.PushBack(std::move(v));
    }
    return json_data.ExtractValue();
  }
}  // namespace Frs
