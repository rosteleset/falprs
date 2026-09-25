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
                            default_config = '{"barcode-confidence":0.8,"best-quality-interval-after":"2s","best-quality-interval-before":"5s","blur":300,"blur-max":13000,"capture-timeout":"2s","delay-after-error":"30s","delay-between-frames":"1s","dnn-bd-inference-server":"127.0.0.1:8001","dnn-fc-inference-server":"127.0.0.1:8001","dnn-fd-inference-server":"127.0.0.1:8001","dnn-fr-inference-server":"127.0.0.1:8001","face-class-confidence":0.7,"face-confidence":0.7,"face-enlarge-scale":1.5,"face-iou-threshold":0.4,"flag-process-barcodes":false,"flag-process-faces":true,"flag-spawned-descriptors":false,"logs-level":"info","margin":5,"max-capture-error-count":3,"open-door-duration":"10s","osd-datetime-format":"%Y-%m-%d %H:%M:%S","title-height-ratio":0.033,"tolerance":0.5,"unknown-descriptor-ttl":"5s","workflow-timeout":"0s"}'
                        else:
                            default_config = '{"ban-duration":"30s","ban-duration-area":"12h","ban-iou-threshold":0.5,"callback-timeout":"2s","capture-timeout":"2s","char-iou-threshold":0.7,"char-score":0.3,"delay-after-error":"30s","delay-between-frames":"1s","event-log-after":"5s","event-log-before":"10s","flag-process-special":false,"flag-save-failed":false,"inference-timeout":"1s","logs-level":"info","lpc-net-inference-server":"127.0.0.1:8001","lpc-net-input-height":224,"lpc-net-input-tensor-name":"input","lpc-net-input-width":224,"lpc-net-model-name":"lpcnet_vit","lpc-net-output-tensor-name":"output","lpd-net-inference-server":"127.0.0.1:8001","lpd-net-input-height":640,"lpd-net-input-tensor-name":"images","lpd-net-input-width":640,"lpd-net-model-name":"lpdnet_yolo","lpd-net-output-tensor-name":"output0","lpr-net-inference-server":"127.0.0.1:8001","lpr-net-input-height":320,"lpr-net-input-tensor-name":"images","lpr-net-input-width":320,"lpr-net-model-name":"lprnet_yolo","lpr-net-output-tensor-name":"output0","max-capture-error-count":3,"min-plate-height":0,"plate-confidence":0.5,"special-confidence":0.9,"vc-net-inference-server":"127.0.0.1:8001","vc-net-input-height":224,"vc-net-input-tensor-name":"input","vc-net-input-width":224,"vc-net-model-name":"vcnet_vit","vc-net-output-tensor-name":"output","vd-net-inference-server":"127.0.0.1:8001","vd-net-input-height":640,"vd-net-input-tensor-name":"images","vd-net-input-width":640,"vd-net-model-name":"vdnet_yolo","vd-net-output-tensor-name":"output0","vehicle-area-ratio-threshold":0.01,"vehicle-confidence":0.7,"vehicle-iou-threshold":0.45,"workflow-timeout":"0s"}'
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
