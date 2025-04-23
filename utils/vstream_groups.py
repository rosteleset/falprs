import psycopg2
import argparse
import yaml
from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', metavar='<path>', default='/opt/falprs/config.yaml', help="path to FALPRS configuration file (default: %(default)s)")
parser.add_argument('-t', '--type', choices=['frs', 'lprs'], required=True, help='project type (required)')
parser.add_argument('-l', '--list', action='store_true', default=False, help='list video stream groups')
parser.add_argument('-a', '--add', metavar='<group_name>', help='add new video stream group')
parser.add_argument('-r', '--remove', metavar='<id_group>', help='remove video stream group and all its data')
args = parser.parse_args()

try:
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
        pg_database = f"{args.type}-postgresql-database"
        if 'dbconnection' in config['components_manager']['components'][pg_database]:
            dbconn = config['components_manager']['components'][pg_database]['dbconnection']
            try:
                pg_conn = psycopg2.connect(dbconn)
                with pg_conn.cursor() as pg_cursor:
                    if args.list:
                        pg_cursor.execute("select id_group, group_name, auth_token from vstream_groups order by 1")
                        table = PrettyTable(['id_group', 'group_name', 'auth_token'])
                        for (id_group, group_name, auth_token) in pg_cursor:
                            table.add_row([id_group, group_name, auth_token])
                        print(table)
                    elif args.add is not None:
                        pg_cursor.execute(f"insert into vstream_groups(group_name, auth_token) values ('{args.add}', uuid_generate_v4()) returning id_group, group_name, auth_token")
                        row = pg_cursor.fetchone()
                        id_group = row[0]
                        group_name = row[1]
                        auth_token = row[2]
                        if args.type == 'frs':
                            default_config = '{"blur": 300.0, "margin": 5.0, "blur-max": 13000.0, "tolerance": 0.5, "title-height-ratio": 0.033, "osd-dt-format": "%Y-%m-%d %H:%M:%S", "logs-level": "info", "capture-timeout": "2s", "face-confidence": 0.7, "delay-after-error": "30s", "face-enlarge-scale": 1.5, "open-door-duration": "10s", "delay-between-frames": "1s", "face-class-confidence": 0.7, "dnn-fc-inference-server": "127.0.0.1:8000", "dnn-fd-inference-server": "127.0.0.1:8000", "dnn-fr-inference-server": "127.0.0.1:8000", "max-capture-error-count": 3, "best-quality-interval-after": "2s", "best-quality-interval-before": "5s"}'
                        else:
                            default_config = '{"char-score": 0.4, "logs-level": "info", "ban-duration": "30s", "capture-timeout": "2s", "event-log-after": "5s", "callback-timeout": "2s", "event-log-before": "10s", "plate-confidence": 0.5, "ban-duration-area": "12h", "ban-iou-threshold": 0.5, "delay-after-error": "30s", "min-plate-height": 0, "flag-save-failed": false, "flag-process-special": false, "char-iou-threshold": 0.7, "lpd-net-model-name": "lpdnet_yolo", "lpr-net-model-name": "lprnet_yolo", "vehicle-confidence": 0.7, "special-confidence": 0.9, "lpd-net-input-width": 640, "lpr-net-input-width": 160, "delay-between-frames": "1s", "lpd-net-input-height": 640, "lpr-net-input-height": 160, "vehicle-iou-threshold": 0.45, "max-capture-error-count": 3, "vd-net-model-name": "vdnet_yolo", "lpd-net-inference-server": "127.0.0.1:8000", "lpr-net-inference-server": "127.0.0.1:8000", "vd-net-input-width": 640, "lpd-net-input-tensor-name": "images", "lpr-net-input-tensor-name": "images", "lpr-net-output-tensor-name": "output0", "vd-net-input-height": 640, "lpd-net-output-tensor-name": "output0", "vehicle-area-ratio-threshold": 0.01, "vd-net-inference-server": "127.0.0.1:8000", "vd-net-input-tensor-name": "images", "vd-net-output-tensor-name": "output0", "vc-net-inference-server": "127.0.0.1:8000", "vc-net-model-name": "vcnet_vit", "vc-net-input-width": 224, "vc-net-input-height": 224, "vc-net-input-tensor-name": "input", "vc-net-output-tensor-name": "output"}'
                        query = f"insert into default_vstream_config (id_group, config) VALUES ({id_group}, '{default_config}') on conflict do nothing"
                        pg_cursor.execute(query)
                        table = PrettyTable(['id_group', 'group_name', 'auth_token'])
                        table.add_row([id_group, group_name, auth_token])
                        print("New group created:")
                        print(table)
                    elif args.remove is not None:
                        answer = input("Attention! This action will delete the video stream group and all data associated with it. Continue (y/n)? ")
                        if answer.lower() in ["y","yes"]:
                            pg_cursor.execute(f"delete from vstream_groups where id_group = {args.remove}")
                            print("Group deleted.")
                    else:
                        parser.print_help()
                pg_conn.commit()
                pg_cursor.close()
            except Exception as error:
                print(error)
        else:
            print(f"Error parsing YAML configuration file: {args.config}")
except Exception as error:
    print(error)
