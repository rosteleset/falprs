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
insert into default_vstream_config (id_group, config) VALUES ((select id_group from vstream_groups where group_name = 'default'), '{"barcode-confidence":0.8,"best-quality-interval-after":"2s","best-quality-interval-before":"5s","blur":300,"blur-max":13000,"capture-timeout":"2s","delay-after-error":"30s","delay-between-frames":"1s","dnn-bd-inference-server":"127.0.0.1:8000","dnn-fc-inference-server":"127.0.0.1:8000","dnn-fd-inference-server":"127.0.0.1:8000","dnn-fr-inference-server":"127.0.0.1:8000","face-class-confidence":0.7,"face-confidence":0.7,"face-enlarge-scale":1.5,"face-iou-threshold":0.4,"flag-process-barcodes":false,"flag-process-faces":true,"flag-spawned-descriptors":false,"logs-level":"info","margin":5,"max-capture-error-count":3,"open-door-duration":"10s","osd-datetime-format":"%Y-%m-%d %H:%M:%S","title-height-ratio":0.033,"tolerance":0.5,"unknown-descriptor-ttl":"5s","workflow-timeout":"0s"}') on conflict do nothing;
