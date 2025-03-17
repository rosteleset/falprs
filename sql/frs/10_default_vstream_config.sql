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
insert into default_vstream_config (id_group, config) VALUES ((select id_group from vstream_groups where group_name = 'default'), '{"blur": 300.0, "margin": 5.0, "blur-max": 13000.0, "tolerance": 0.5, "title-height-ratio": 0.033, "osd-dt-format": "%Y-%m-%d %H:%M:%S", "logs-level": "info", "capture-timeout": "2s", "face-confidence": 0.7, "delay-after-error": "30s", "face-enlarge-scale": 1.5, "open-door-duration": "10s", "delay-between-frames": "1s", "face-class-confidence": 0.7, "dnn-fc-inference-server": "127.0.0.1:8000", "dnn-fd-inference-server": "127.0.0.1:8000", "dnn-fr-inference-server": "127.0.0.1:8000", "max-capture-error-count": 3, "best-quality-interval-after": "2s", "best-quality-interval-before": "5s"}') on conflict do nothing;
