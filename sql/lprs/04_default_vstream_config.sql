create table if not exists default_vstream_config
(
  id_group integer not null
    constraint default_vstream_one_row
      primary key
    constraint default_vstream_config_vstream_groups_id_group_fk
      references vstream_groups
      on update cascade on delete cascade,
  config   jsonb
);
comment on table default_vstream_config is 'Default video stream configuration';
insert into default_vstream_config (id_group, config) values ((select id_group from vstream_groups where group_name = 'default'), '{"ban-duration":"30s","ban-duration-area":"12h","ban-iou-threshold":0.5,"callback-timeout":"2s","capture-timeout":"2s","char-iou-threshold":0.7,"char-score":0.3,"delay-after-error":"30s","delay-between-frames":"1s","event-log-after":"5s","event-log-before":"10s","flag-process-special":false,"flag-save-failed":false,"inference-timeout":"1s","logs-level":"info","lpc-net-inference-server":"127.0.0.1:8000","lpc-net-input-height":224,"lpc-net-input-tensor-name":"input","lpc-net-input-width":224,"lpc-net-model-name":"lpcnet_vit","lpc-net-output-tensor-name":"output","lpd-net-inference-server":"127.0.0.1:8000","lpd-net-input-height":640,"lpd-net-input-tensor-name":"images","lpd-net-input-width":640,"lpd-net-model-name":"lpdnet_yolo","lpd-net-output-tensor-name":"output0","lpr-net-inference-server":"127.0.0.1:8000","lpr-net-input-height":320,"lpr-net-input-tensor-name":"images","lpr-net-input-width":320,"lpr-net-model-name":"lprnet_yolo","lpr-net-output-tensor-name":"output0","max-capture-error-count":3,"min-plate-height":0,"plate-confidence":0.5,"special-confidence":0.9,"vc-net-inference-server":"127.0.0.1:8000","vc-net-input-height":224,"vc-net-input-tensor-name":"input","vc-net-input-width":224,"vc-net-model-name":"vcnet_vit","vc-net-output-tensor-name":"output","vd-net-inference-server":"127.0.0.1:8000","vd-net-input-height":640,"vd-net-input-tensor-name":"images","vd-net-input-width":640,"vd-net-model-name":"vdnet_yolo","vd-net-output-tensor-name":"output0","vehicle-area-ratio-threshold":0.01,"vehicle-confidence":0.7,"vehicle-iou-threshold":0.45,"workflow-timeout":"0s"}') on conflict do nothing;
update default_vstream_config set config = config || '{"lpc-net-inference-server":"127.0.0.1:8000","lpc-net-input-height":224,"lpc-net-input-tensor-name":"input","lpc-net-input-width":224,"lpc-net-model-name":"lpcnet_vit","lpc-net-output-tensor-name":"output","lpd-net-inference-server":"127.0.0.1:8000","lpr-net-input-height":320,"lpr-net-input-width":320}';
