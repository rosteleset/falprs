#pragma once

#include <absl/strings/string_view.h>
#include <absl/strings/substitute.h>
#include <immintrin.h>
#include <userver/server/handlers/http_handler_json_base.hpp>
#include <userver/storages/postgres/cluster.hpp>

#include "frs_caches.hpp"
#include "frs_workflow.hpp"

namespace Frs
{
  constexpr int DESCRIPTOR_SIZE = 512;
  typedef float Data[DESCRIPTOR_SIZE];

  // binary event data
  struct EventData
  {
    char event_id[32];  // internal event identifier
    int32_t position;   // descriptor position (numbering starts from zero)
    Data data;          // descriptor data
  } __attribute__((packed));

  struct DescriptorData
  {
    Data data;  // descriptor data
  } __attribute__((packed));

  struct ResultItem
  {
    std::string event_date;
    std::string event_id;   // internal event identifier
    std::string uuid;       // host event identifier
    std::string url_image;  // frame URL
    int id_descriptor = 0;  // descriptor identifier
    double similarity = -1.0;

    bool operator>(const ResultItem& other) const
    {
      return event_date > other.event_date;
    }
  };

  inline float reduceSum(const __m256& a)
  {
    __m256 s0 = _mm256_hadd_ps(a, a);
    s0 = _mm256_hadd_ps(s0, s0);
    auto s1 = _mm256_extractf128_ps(s0, 1);
    s1 = _mm_add_ps(_mm256_castps256_ps128(s0), s1);
    return _mm_cvtss_f32(s1);
  }

  inline double cosineDistanceSIMD(const Data& d1, const Data& d2)
  {
    constexpr int step = 8;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum0_sqr1 = _mm256_setzero_ps();
    __m256 sum1_sqr1 = _mm256_setzero_ps();
    __m256 sum2_sqr1 = _mm256_setzero_ps();
    __m256 sum3_sqr1 = _mm256_setzero_ps();
    __m256 sum0_sqr2 = _mm256_setzero_ps();
    __m256 sum1_sqr2 = _mm256_setzero_ps();
    __m256 sum2_sqr2 = _mm256_setzero_ps();
    __m256 sum3_sqr2 = _mm256_setzero_ps();

    for (int i = 0; i < DESCRIPTOR_SIZE; i += 4 * step)
    {
      __m256 a0 = _mm256_loadu_ps(d1 + i + 0 * step);
      __m256 b0 = _mm256_loadu_ps(d2 + i + 0 * step);
      sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, b0));
      sum0_sqr1 = _mm256_add_ps(sum0_sqr1, _mm256_mul_ps(a0, a0));
      sum0_sqr2 = _mm256_add_ps(sum0_sqr2, _mm256_mul_ps(b0, b0));

      a0 = _mm256_loadu_ps(d1 + i + 1 * step);
      b0 = _mm256_loadu_ps(d2 + i + 1 * step);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a0, b0));
      sum1_sqr1 = _mm256_add_ps(sum1_sqr1, _mm256_mul_ps(a0, a0));
      sum1_sqr2 = _mm256_add_ps(sum1_sqr2, _mm256_mul_ps(b0, b0));

      a0 = _mm256_loadu_ps(d1 + i + 2 * step);
      b0 = _mm256_loadu_ps(d2 + i + 2 * step);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a0, b0));
      sum2_sqr1 = _mm256_add_ps(sum2_sqr1, _mm256_mul_ps(a0, a0));
      sum2_sqr2 = _mm256_add_ps(sum2_sqr2, _mm256_mul_ps(b0, b0));

      a0 = _mm256_loadu_ps(d1 + i + 3 * step);
      b0 = _mm256_loadu_ps(d2 + i + 3 * step);
      sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a0, b0));
      sum3_sqr1 = _mm256_add_ps(sum3_sqr1, _mm256_mul_ps(a0, a0));
      sum3_sqr2 = _mm256_add_ps(sum3_sqr2, _mm256_mul_ps(b0, b0));
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    sum0_sqr1 = _mm256_add_ps(sum0_sqr1, sum1_sqr1);
    sum2_sqr1 = _mm256_add_ps(sum2_sqr1, sum3_sqr1);
    sum0_sqr1 = _mm256_add_ps(sum0_sqr1, sum2_sqr1);
    sum0_sqr2 = _mm256_add_ps(sum0_sqr2, sum1_sqr2);
    sum2_sqr2 = _mm256_add_ps(sum2_sqr2, sum3_sqr2);
    sum0_sqr2 = _mm256_add_ps(sum0_sqr2, sum2_sqr2);
    return reduceSum(sum0) / sqrt(reduceSum(sum0_sqr1)) / sqrt(reduceSum(sum0_sqr2));
  }

  class Api final : public userver::server::handlers::HttpHandlerJsonBase
  {
  public:
    static constexpr auto kName = "frs-api-http";

    // API methods
    static constexpr auto METHOD_ADD_STREAM = "addStream";                              // add or change video stream
    static constexpr auto METHOD_MOTION_DETECTION = "motionDetection";                  // provide motion detection info
    static constexpr auto METHOD_DOOR_IS_OPEN = "doorIsOpen";                           // notify that door is open
    static constexpr auto METHOD_BEST_QUALITY = "bestQuality";                          // get best face info
    static constexpr auto METHOD_GET_EVENTS = "getEvents";                              // get a list of events from a time interval
    static constexpr auto METHOD_REGISTER_FACE = "registerFace";                        // register face
    static constexpr auto METHOD_ADD_FACES = "addFaces";                                // attach face descriptor to video stream
    static constexpr auto METHOD_REMOVE_FACES = "removeFaces";                          // detach face descriptor from video stream
    static constexpr auto METHOD_LIST_STREAMS = "listStreams";                          // list video streams with attached faces
    static constexpr auto METHOD_REMOVE_STREAM = "removeStream";                        // remove video stream
    static constexpr auto METHOD_LIST_ALL_FACES = "listAllFaces";                       // list all faces
    static constexpr auto METHOD_DELETE_FACES = "deleteFaces";                          // delete faces from database (regardless of attaching to video streams)
    static constexpr auto METHOD_TEST_IMAGE = "testImage";                              // test image
    static constexpr auto METHOD_PROCESS_FRAME = "processFrame";                        // process frame by url
    static constexpr auto METHOD_ADD_SPECIAL_GROUP = "addSpecialGroup";                 // add special group
    static constexpr auto METHOD_UPDATE_SPECIAL_GROUP = "updateSpecialGroup";           // update special group
    static constexpr auto METHOD_DELETE_SPECIAL_GROUP = "deleteSpecialGroup";           // delete special group
    static constexpr auto METHOD_LIST_SPECIAL_GROUPS = "listSpecialGroups";             // list special groups
    static constexpr auto SG_METHOD_PREFIX = "sg";                                      // prefix for special group methods
    static constexpr auto METHOD_SG_REGISTER_FACE = "sgRegisterFace";                   // register a person in a special group
    static constexpr auto METHOD_SG_DELETE_FACES = "sgDeleteFaces";                     // remove the list of special group descriptors from the database
    static constexpr auto METHOD_SG_LIST_FACES = "sgListFaces";                         // get a list of all special group face descriptors
    static constexpr auto METHOD_SG_UPDATE_GROUP = "sgUpdateGroup";                     // update special group parameters
    static constexpr auto METHOD_SG_RENEW_TOKEN = "sgRenewToken";                       // renew special group authorization Token
    static constexpr auto METHOD_SG_SEARCH_FACES = "sgSearchFaces";                     // search faces in special group
    static constexpr auto METHOD_SAVE_DNN_STATS_DATA = "saveDnnStatsData";              // save inference statistics data
    static constexpr auto METHOD_SET_COMMON_CONFIG = "setCommonConfig";                 // set common configuration parameters
    static constexpr auto METHOD_GET_COMMON_CONFIG = "getCommonConfig";                 // get common configuration parameters
    static constexpr auto METHOD_SET_STREAM_DEFAULT_CONFIG = "setStreamDefaultConfig";  // set common configuration parameters
    static constexpr auto METHOD_GET_STREAM_DEFAULT_CONFIG = "getStreamDefaultConfig";  // get common configuration parameters

    // parameters
    static constexpr auto P_CODE = "code";
    static constexpr auto P_MESSAGE = "message";

    static constexpr auto P_DATA = "data";
    static constexpr auto P_STREAM_ID = "streamId";
    static constexpr auto P_URL = "url";
    static constexpr auto P_FACE_IDS = "faces";
    static constexpr auto P_CALLBACK_URL = "callback";
    static constexpr auto P_START = "start";
    static constexpr auto P_DATE = "date";
    static constexpr auto P_LOG_EVENT_ID = "eventId";
    static constexpr auto P_EVENT_UUID = "uuid";
    static constexpr auto P_SCREENSHOT_URL = "screenshot";
    static constexpr auto P_FACE_LEFT = "left";
    static constexpr auto P_FACE_TOP = "top";
    static constexpr auto P_FACE_WIDTH = "width";
    static constexpr auto P_FACE_HEIGHT = "height";
    static constexpr auto P_FACE_ID = "faceId";
    static constexpr auto P_FACE_IMAGE = "faceImage";
    static constexpr auto P_CONFIG = "config";
    static constexpr auto P_PARAMS = "params";  // for compatibility with old API
    static constexpr auto P_QUALITY = "quality";
    static constexpr auto P_DATE_START = "dateStart";
    static constexpr auto P_DATE_END = "dateEnd";
    static constexpr auto P_SPECIAL_GROUP_NAME = "groupName";
    static constexpr auto P_MAX_DESCRIPTOR_COUNT = "maxDescriptorCount";
    static constexpr auto P_SG_API_TOKEN = "accessApiToken";
    static constexpr auto P_SG_ID = "groupId";
    static constexpr auto P_SEARCH_IN_LOGS = "useLogs";
    static constexpr auto P_SEARCH_IN_EVENTS = "useEvents";
    static constexpr auto P_SIMILARITY = "similarity";
    static constexpr auto P_SIMILARITY_THRESHOLD = "similarityThreshold";

    // messages
    static constexpr auto MESSAGE_REQUEST_COMPLETED = "Request completed successfully";

    // errors
    static constexpr auto ERROR_UNKNOWN_METHOD = "Unknown API method";

    // queries
    static constexpr auto SQL_GET_VSTREAM_ID = "select id_vstream from video_streams where id_group = $1 and vstream_ext = $2 and not flag_deleted";

    static constexpr auto SQL_GET_STREAM = R"__SQL__(
      select
        id_vstream,
        url,
        callback_url
      from
        video_streams
      where
        id_group = $1
        and vstream_ext = $2
    )__SQL__";

    static constexpr auto SQL_ADD_STREAM = R"__SQL__(
      insert into video_streams(id_group, vstream_ext, url, callback_url, config) values($1, $2, $3, $4, $5) returning id_vstream
    )__SQL__";

    static constexpr auto SQL_UPDATE_STREAM = R"__SQL__(
      update
        video_streams
      set
        url = $2,
        callback_url = $3,
        flag_deleted = false,
        config = $4,
        last_updated = now()
      where
        id_group = $1
        and id_vstream = $5
    )__SQL__";

    static constexpr auto SQL_ADD_LINK_DESCRIPTOR_VSTREAM = R"__SQL__(
      insert into link_descriptor_vstream(id_vstream, id_descriptor)
          select $2, f.id_descriptor from face_descriptors f where f.id_descriptor = $3 and f.id_group = $1
      on conflict (id_vstream, id_descriptor) do update set last_updated = now(), flag_deleted = false
    )__SQL__";

    // do not delete row from database, just mark for deletion
    static constexpr auto SQL_REMOVE_LINK_DESCRIPTOR_VSTREAM =
      "update link_descriptor_vstream set last_updated = now(), flag_deleted = true where id_vstream = $1 and id_descriptor = $2";

    // do not delete rows from database, just mark for deletion
    static constexpr auto SQL_REMOVE_LINK_DESCRIPTOR_VSTREAM_BY_VSTREAM =
      "update link_descriptor_vstream set last_updated = now(), flag_deleted = true where id_vstream = $1";

    // do not delete descriptor from database, just mark for deletion
    static constexpr auto SQL_REMOVE_DESCRIPTOR =
      "update face_descriptors set last_updated = now(), flag_deleted = true where id_group = $1 and id_descriptor = $2";

    // do not delete rows from database, just mark for deletion
    static constexpr auto SQL_REMOVE_LINK_DESCRIPTOR_VSTREAM_BY_DESCRIPTOR = R"__SQL__(
      update
        link_descriptor_vstream
      set
        last_updated = now(),
        flag_deleted = true
      where
        id_descriptor = $2
        and $1 = (select fd.id_group from face_descriptors fd where fd.id_descriptor = $2)
    )__SQL__";

    static constexpr auto SQL_LIST_STREAMS_SIMPLE = R"__SQL__(
      select
        v.vstream_ext,
        v.url,
        v.callback_url,
        v.config
      from
        video_streams v
      where
        v.id_group = $1
        and not v.flag_deleted
    )__SQL__";

    static constexpr auto SQL_LIST_STREAM_FACES = R"__SQL__(
      select
        v.vstream_ext,
        ldv.id_descriptor
      from
        video_streams v
        inner join link_descriptor_vstream ldv
          on v.id_vstream = ldv.id_vstream
          and not ldv.flag_deleted
      where
        v.id_group = $1
        and not v.flag_deleted
    )__SQL__";

    static constexpr auto SQL_GET_LOG_FACE_BEST_QUALITY = R"_SQL_(
      select
        l.id_log,
        l.screenshot_url,
        l.face_left,
        l.face_top,
        l.face_width,
        l.face_height,
        l.log_date,
        l.copy_data
      from
        log_faces l
      where
        l.id_vstream = $0
        and (l.log_date >= timestamptz '$1' - interval '$2 millisecond') and (l.log_date <= timestamptz '$1' + interval '$3 millisecond')
        and l.copy_data >= 0
      order by
        l.quality desc
      limit
        1
    )_SQL_";

    static constexpr auto SQL_GET_LOG_FACE_BY_ID = R"_SQL_(
      select
        l.id_log,
        l.screenshot_url,
        l.face_left,
        l.face_top,
        l.face_width,
        l.face_height,
        l.log_date,
        l.copy_data
      from
        log_faces l
        join video_streams vs
          on vs.id_vstream = l.id_vstream
      where
        vs.id_group = $1
        and l.id_log = $2
    )_SQL_";

    static constexpr auto SQL_GET_LOG_FACES_FROM_INTERVAL = R"_SQL_(
      select
        l.id_log,
        l.log_date,
        l.id_descriptor,
        l.quality,
        l.screenshot_url,
        l.face_left,
        l.face_top,
        l.face_width,
        l.face_height
      from
        log_faces l
      where
        l.id_vstream = $0
        and l.log_date >= '$1'
        and l.log_date <= '$2'
      order by
        l.log_date
    )_SQL_";

    static constexpr auto SQL_SET_COPY_DATA_BY_ID = R"_SQL_(
      update
        log_faces
      set
        copy_data = $1,
        ext_event_uuid = $2
      where
        id_log = $3
    )_SQL_";

    static constexpr auto SQL_DELETE_VIDEO_STREAM = R"_SQL_(
      update
        video_streams
      set
        last_updated = now(),
        flag_deleted = true
      where
        id_group = $1
        and id_vstream = $2
    )_SQL_";

    static constexpr auto SQL_LIST_ALL_FACES = R"_SQL_(
      select
        id_descriptor
      from
        face_descriptors
      where
        id_group = $1
        and not flag_deleted
        and id_descriptor not in (select ldsg.id_descriptor from link_descriptor_sgroup ldsg)
    )_SQL_";

    static constexpr auto SQL_ADD_SPECIAL_GROUP = R"_SQL_(
      insert into
        special_groups(id_group, group_name, sg_api_token, max_descriptor_count)
      values($1, $2, gen_random_uuid(), $3) returning id_special_group, sg_api_token
    )_SQL_";

    static constexpr auto SQL_UPDATE_SPECIAL_GROUP = R"_SQL_(
      update
        special_groups
      set
        group_name = $0,
        max_descriptor_count = $1
      where
        id_group = $2
        and id_special_group = $3
    )_SQL_";

    static constexpr auto SQL_DELETE_SPECIAL_GROUP = R"_SQL_(
      update
        special_groups
      set
        last_updated = now(),
        flag_deleted = true
      where
        id_group = $1
        and id_special_group = $2
    )_SQL_";

    static constexpr auto SQL_LIST_SPECIAL_GROUPS = R"_SQL_(
      select
        id_special_group,
        group_name,
        sg_api_token,
        callback_url,
        max_descriptor_count
      from
        special_groups
      where
        id_group = $1
        and flag_deleted = false
    )_SQL_";

    static constexpr auto SQL_SET_COMMON_CONFIG_PARAMS = "update common_config set config = coalesce(config, $2) || $2 where id_group = $1";
    static constexpr auto SQL_GET_COMMON_CONFIG_PARAMS = "select config from common_config where id_group = $1";
    static constexpr auto SQL_SET_STREAM_DEFAULT_CONFIG_PARAMS = "update default_vstream_config set config = coalesce(config, $2) || $2 where id_group = $1";
    static constexpr auto SQL_GET_STREAM_DEFAULT_CONFIG_PARAMS = "select config from default_vstream_config where id_group = $1";

    // do not delete row from database, just mark for deletion
    static constexpr auto SQL_REMOVE_LINK_DESCRIPTOR_SG =
      "update link_descriptor_sgroup set last_updated = now(), flag_deleted = true where id_sgroup = $1 and id_descriptor = $2";

    // do not delete rows from database, just mark for deletion
    static constexpr auto SQL_REMOVE_LINK_DESCRIPTOR_SG_ALL =
      "update link_descriptor_sgroup set last_updated = now(), flag_deleted = true where id_sgroup = $1";

    // do not delete descriptor from database, just mark for deletion
    static constexpr auto SQL_REMOVE_SG_FACE_DESCRIPTOR = R"_SQL_(
      update
        face_descriptors fd
      set
        last_updated = now(),
        flag_deleted = true
      where
        id_descriptor = $2
        and id_descriptor in (select ldsg.id_descriptor from link_descriptor_sgroup ldsg where ldsg.id_sgroup = $1)
    )_SQL_";

    // do not delete descriptor from database, just mark for deletion
    static constexpr auto SQL_REMOVE_SG_FACE_DESCRIPTORS = R"_SQL_(
      update
        face_descriptors fd
      set
        last_updated = now(),
        flag_deleted = true
      where
        id_descriptor in (select ldsg.id_descriptor from link_descriptor_sgroup ldsg where ldsg.id_sgroup = $1)
    )_SQL_";

    static constexpr auto SQL_SG_LIST_FACES = R"_SQL_(
      select
        di.id_descriptor,
        concat('data:', di.mime_type, ';base64,', translate(encode(di.face_image, 'base64'), E'\n', '')) face_image
      from
        link_descriptor_sgroup ldsg
        inner join face_descriptors fd
          on ldsg.id_descriptor = fd.id_descriptor
        inner join descriptor_images di
          on fd.id_descriptor = di.id_descriptor
      where
        ldsg.id_sgroup = $1
        and not ldsg.flag_deleted
        and not fd.flag_deleted
    )_SQL_";

    static constexpr auto SQL_SG_UPDATE_GROUP = R"_SQL_(
      update
        special_groups
      set
        set callback_url = $2
      where
        id_special_group = $1
    )_SQL_";

    static constexpr auto SQL_SG_RENEW_TOKEN = R"_SQL_(
      update
        special_groups
      set
        sg_api_token = gen_random_uuid()
      where
        id_special_group = $1
      returning sg_api_token
    )_SQL_";

    static constexpr auto SQL_SG_DESCRIPTORS = R"_SQL_(
      select
        f.id_descriptor,
        f.descriptor_data
      from
        link_descriptor_sgroup l
        inner join face_descriptors f
        on f.id_descriptor = l.id_descriptor
      where
        not l.flag_deleted
        and l.id_sgroup = $0
        and f.id_descriptor in ($1)
    )_SQL_";

    // Component is valid after construction and is able to accept requests
    Api(const userver::components::ComponentConfig& config, const userver::components::ComponentContext& context);

    userver::formats::json::Value HandleRequestJsonThrow(const userver::server::http::HttpRequest&,
      const userver::formats::json::Value& json,
      userver::server::request::RequestContext&) const override;

  private:
    userver::storages::postgres::ClusterPtr pg_cluster_;
    const GroupsCache& groups_cache_;
    const ConfigCache& config_cache_;
    const VStreamsConfigCache& vstreams_config_cache_;
    const SGConfigCache& sg_config_cache_;
    Workflow& workflow_;

    int32_t checkToken(absl::string_view token) const;
    int32_t checkSGToken(absl::string_view token) const;
    int32_t getVStreamId(int32_t id_group, absl::string_view vstream_ext) const;
    static void requireMemberThrow(const userver::formats::json::Value& json, absl::string_view member);
    static void requireArrayThrow(const userver::formats::json::Value& json, absl::string_view member);
    void addStream(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value listStreams(int32_t id_group) const;
    void motionDetection(int32_t id_group, const userver::formats::json::Value& json) const;
    void doorIsOpen(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value bestQuality(int32_t id_group, const userver::formats::json::Value& json) const;
    void addFaces(int32_t id_group, const userver::formats::json::Value& json) const;
    void removeFaces(int32_t id_group, const userver::formats::json::Value& json) const;
    void removeVStream(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value listAllFaces(int32_t id_group) const;
    void deleteFaces(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value getEvents(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value registerFace(int32_t id_group, const userver::formats::json::Value& json) const;
    void testImage(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value processFrame(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value addSpecialGroup(int32_t id_group, const userver::formats::json::Value& json) const;
    void updateSpecialGroup(int32_t id_group, const userver::formats::json::Value& json) const;
    void deleteSpecialGroup(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value listSpecialGroups(int32_t id_group) const;
    void saveDNNStatsData() const;
    void setCommonConfigParams(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value getCommonConfigParams(int32_t id_group) const;
    void setStreamDefaultConfigParams(int32_t id_group, const userver::formats::json::Value& json) const;
    userver::formats::json::Value getStreamDefaultConfigParams(int32_t id_group) const;

    // member functions for special group methods
    userver::formats::json::Value sgRegisterFace(int32_t id_sgroup, const userver::formats::json::Value& json) const;
    void sgDeleteFaces(int32_t id_sgroup, const userver::formats::json::Value& json) const;
    userver::formats::json::Value sgListFaces(int32_t id_sgroup) const;
    void sgUpdateGroup(int32_t id_sgroup, const userver::formats::json::Value& json) const;
    userver::formats::json::Value sgRenewToken(int32_t id_sgroup) const;
    userver::formats::json::Value sgSearchFaces(int32_t id_sgroup, const userver::formats::json::Value& json) const;
  };
}  // namespace Frs
