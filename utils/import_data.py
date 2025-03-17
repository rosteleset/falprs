from mysql.connector import connect
import psycopg2
import json
import os
import shutil

from config import *

int_params_common = [
    'dnn-fd-input-width',
    'dnn-fd-input-height',
    'dnn-fc-input-height',
    'dnn-fc-input-width',
    'dnn-fc-output-size',
    'dnn-fr-input-width',
    'dnn-fr-input-height',
    'dnn-fr-output-size',
    'sg-max-descriptor-count',
]

int_params_vstream = [
    'max-capture-error-count',
]

double_params_vstream = [
    'alpha',
    'best-quality-interval-after',
    'best-quality-interval-before',
    'beta',
    'blur',
    'blur-max',
    'face-class-confidence',
    'face-confidence',
    'face-enlarge-scale',
    'gamma',
    'margin',
    'tolerance',
]

string_params_common = [
    'dnn-fd-model-name',
    'dnn-fd-input-tensor-name',
    'dnn-fc-model-name',
    'dnn-fc-input-tensor-name',
    'dnn-fc-output-tensor-name',
    'dnn-fr-model-name',
    'dnn-fr-input-tensor-name',
    'dnn-fr-output-tensor-name',
    'comments-blurry-face',
    'comments-descriptor-creation-error',
    'comments-descriptor-exists',
    'comments-inference-error',
    'comments-new-descriptor',
    'comments-no-faces',
    'comments-non-frontal-face',
    'comments-non-normal-face-class',
    'comments-partial-face',
    'comments-url-image-error',
]

string_params_vstream = [
    'dnn-fd-inference-server',
    'dnn-fc-inference-server',
    'dnn-fr-inference-server',
]

duration_params_common = [
    'callback-timeout',
]

duration_params_vstream = [
    'best-quality-interval-after',
    'best-quality-interval-before',
    'capture-timeout',
    'delay-after-error',
    'delay-between-frames',
    'open-door-duration',
    'retry-pause',
]

bool_params_common = [
    'flag-copy-event-data'
]

map_logs_level = {
    '0': 'error',
    '1': 'info',
    '2': 'trace',
}

try:
    pg_conn = psycopg2.connect(dbname=pg_database, user=pg_user,
                               password=pg_password, host=pg_host, port=pg_port)
    with pg_conn.cursor() as pg_cursor:
        pg_cursor.execute("select id_group from vstream_groups where group_name = 'default'")
        id_group = pg_cursor.fetchone()[0]
        print('default group =', id_group)
except Exception as error:
    print(error)
    exit(-1)

try:
    mysql_conn = connect(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database
    )
    mysql_conn2 = connect(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database
    )
    # video_streams
    print("Processing video streams...", end="")
    query = "select id_vstream, vstream_ext, url, callback_url, region_x, region_y, region_width, region_height from video_streams"
    with mysql_conn.cursor() as cursor:
        cursor.execute(query)
        seq_max = 0
        for (id_vstream, vstream_ext, url, callback_url, region_x, region_y, region_width, region_height) in cursor:
            if id_vstream > seq_max:
                seq_max = id_vstream
            config = {}
            query = f"select vs.param_name, vs.param_value from video_stream_settings vs where vs.id_vstream = {id_vstream}"
            with mysql_conn2.cursor() as cursor2:
                cursor2.execute(query)
                for (param_name, param_value) in cursor2:
                    if param_name in int_params_vstream:
                        config[param_name] = int(param_value)
                    if param_name in double_params_vstream:
                        config[param_name] = float(param_value)
                    if param_name in string_params_vstream:
                        config[param_name] = param_value
                    if param_name in duration_params_vstream:
                        duration = int(float(param_value) * 1000)
                        config[param_name] = str(duration) + 'ms'
                    if param_name == 'logs-level':
                        config[param_name] = map_logs_level[param_value]
                cursor2.close()
            if region_width > 0 and region_height > 0:
                config["workArea"] = [region_x, region_y, region_width, region_height]
            query = 'insert into video_streams(id_vstream, vstream_ext, url, callback_url, id_group, config) values(%s, %s, %s, %s, %s, %s) on conflict do nothing'
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(query, (id_vstream, vstream_ext, url, callback_url, id_group, json.dumps(config)))
                pg_conn.commit()
                pg_cursor.close()
        cursor.close()
        if seq_max > 0:
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(f"alter sequence video_streams_id_vstream_seq restart {seq_max + 1}")
                pg_conn.commit()
                pg_cursor.close()
    print("done.")

    # face_descriptor, descriptor_images
    print("Processing face descriptors and their images...", end="")
    query = """
        select
          fd.id_descriptor,
          fd.date_start,
          fd.descriptor_data,
          di.face_image,
          di.mime_type
        from
          face_descriptors fd
          inner join descriptor_images di
            on di.id_descriptor = fd.id_descriptor    
    """
    with mysql_conn.cursor() as cursor:
        cursor.execute(query)
        seq_max = 0
        for (id_descriptor, date_start, descriptor_data, face_image, mime_type) in cursor:
            if id_descriptor > seq_max:
                seq_max = id_descriptor
            query = 'insert into face_descriptors(id_descriptor, descriptor_data, date_start, date_last, last_updated, id_group) values (%s, %s, %s, %s, %s, %s) on conflict do nothing'
            query2 = 'insert into descriptor_images(id_descriptor, mime_type, face_image) values(%s, %s, %s) on conflict do nothing'
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(query, (id_descriptor, descriptor_data, date_start, date_start, date_start, id_group))
                pg_cursor.execute(query2, (id_descriptor, mime_type, face_image))
                pg_conn.commit()
                pg_cursor.close()
        cursor.close()
        if seq_max > 0:
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(f"alter sequence face_descriptors_id_descriptor_seq restart {seq_max}")
                pg_conn.commit()
                pg_cursor.close()
    print("done.")

    # link_descriptor_vstream
    print("Processing linked descriptors...", end="")
    query = 'select id_descriptor, id_vstream from link_descriptor_vstream'
    with mysql_conn.cursor() as cursor:
        cursor.execute(query)
        for (id_descriptor, id_vstream) in cursor:
            query = 'insert into link_descriptor_vstream(id_descriptor, id_vstream) values(%s, %s) on conflict do nothing'
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(query, (id_descriptor, id_vstream))
                pg_conn.commit()
                pg_cursor.close()
        cursor.close()
    print("done.")

    # log_faces
    print("Processing log faces...", end="")
    query = 'select id_log, id_vstream, log_date, id_descriptor, quality, screenshot, face_left, face_top, face_width, face_height from log_faces'
    with mysql_conn.cursor() as cursor:
        cursor.execute(query)
        seq_max = 0
        for (id_log, id_vstream, log_date, id_descriptor, quality, screenshot, face_left, face_top, face_width, face_height) in cursor:
            if id_log > seq_max:
                seq_max = id_log
            screenshot_url = screenshots_url_prefix_new + '/' + screenshot[:-44] + 'group_' + str(id_group) + '/' + screenshot[-44:]
            uuid = screenshot[-36:-4]
            log_uuid = uuid[:8] + '-' + uuid[8:12] + '-' + uuid[12:16] + '-' + uuid[16:20] + '-' + uuid[20:]
            query = 'insert into log_faces(id_log, id_vstream, log_date, id_descriptor, quality, face_left, face_top, face_width, face_height, screenshot_url, log_uuid) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing'
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(query, (id_log, id_vstream, log_date, id_descriptor, quality, face_left, face_top, face_width, face_height, screenshot_url, log_uuid))
                pg_conn.commit()
                pg_cursor.close()
        cursor.close()
        if seq_max > 0:
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(f"alter sequence log_faces_id_log_seq restart {seq_max}")
                pg_conn.commit()
                pg_cursor.close()
    print("done.")

    # special groups
    print("Processing special groups...", end="")
    query = 'select id_special_group, group_name, sg_api_token, callback_url, max_descriptor_count from special_groups'
    with mysql_conn.cursor() as cursor:
        cursor.execute(query)
        seq_max = 0
        for (id_special_group, group_name, sg_api_token, callback_url, max_descriptor_count) in cursor:
            if id_special_group > seq_max:
                seq_max = id_special_group
            query = 'insert into special_groups(id_special_group, group_name, sg_api_token, callback_url, max_descriptor_count, id_group) VALUES (%s, %s, %s, %s, %s, %s) on conflict do nothing'
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(query, (id_special_group, group_name, sg_api_token, callback_url, max_descriptor_count, id_group))
                pg_conn.commit()
                pg_cursor.close()
        cursor.close()
        if seq_max > 0:
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(f"alter sequence special_groups_id_special_group_seq restart {seq_max}")
                pg_conn.commit()
                pg_cursor.close()
    print("done.")

    # link_descriptor_sgroup
    print("Processing linked descriptors for special groups...", end="")
    query = 'select id_descriptor, id_sgroup from link_descriptor_sgroup'
    with mysql_conn.cursor() as cursor:
        cursor.execute(query)
        for (id_descriptor, id_sgroup) in cursor:
            query = 'insert into link_descriptor_sgroup(id_descriptor, id_sgroup) values(%s, %s) on conflict do nothing'
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute(query, (id_descriptor, id_sgroup))
                pg_conn.commit()
                pg_cursor.close()
        cursor.close()
    print("done.")

    # config params
    print("Processing configuration parameters...", end="")
    query = "select param_name, param_value from common_settings"
    with mysql_conn.cursor() as cursor:
        config_common = {}
        config_default_vstream = {}
        cursor.execute(query)
        for (param_name, param_value) in cursor:
            if param_name in int_params_common:
                config_common[param_name] = int(param_value)
            if param_name in int_params_vstream:
                config_default_vstream[param_name] = int(param_value)
            if param_name in double_params_vstream:
                config_default_vstream[param_name] = float(param_value)
            if param_name in string_params_common:
                config_common[param_name] = param_value
            if param_name in string_params_vstream:
                config_default_vstream[param_name] = param_value
            if param_name in duration_params_common:
                duration = int(float(param_value) * 1000)
                config_common[param_name] = str(duration) + 'ms'
            if param_name in duration_params_vstream:
                duration = int(float(param_value) * 1000)
                config_default_vstream[param_name] = str(duration) + 'ms'
            if param_name in bool_params_common:
                config_common[param_name] = bool(int(param_value) == 1)
            if param_name == 'logs-level':
                config_default_vstream[param_name] = map_logs_level[param_value]
        cursor.close()
    query = "update common_config set config = coalesce(config, '{}') || %s where id_group = %s"
    with pg_conn.cursor() as pg_cursor:
        pg_cursor.execute(query, (json.dumps(config_common), id_group))
        pg_conn.commit()
        pg_cursor.close()

    query = "update default_vstream_config set config = coalesce(config, '{}') || %s where id_group = %s"
    with pg_conn.cursor() as pg_cursor:
        pg_cursor.execute(query, (json.dumps(config_default_vstream), id_group))
        pg_conn.commit()
        pg_cursor.close()
    print("done.")

    # copy screenshots
    print("Copying screenshots...", end="")
    screenshot_path_new = os.path.join(screenshot_path_new, '') + 'group_' + str(id_group)
    try:
        shutil.copytree(screenshot_path_old, screenshot_path_new, dirs_exist_ok=True)
    except Error as error:
        print(error)
    print("done.")

    # copy events
    print("Copying events data...", end="")
    events_path_new = os.path.join(events_path_new, '') + 'group_' + str(id_group)
    try:
        shutil.copytree(events_path_old, events_path_new, dirs_exist_ok=True)
    except Exception as error:
        print(error)
    print("done.")

    mysql_conn.close()
    mysql_conn2.close()
    pg_conn.close()

    print("Import completed successfully.")
except Exception as error:
    print(error)
    exit(-1)
