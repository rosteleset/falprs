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
insert into default_vstream_config (id_group, config) values (1, '{"char-score": 0.4, "logs-level": "info", "ban-duration": "30s", "capture-timeout": "2s", "event-log-after": "5s", "callback-timeout": "2s", "event-log-before": "10s", "plate-confidence": 0.5, "ban-duration-area": "12h", "ban-iou-threshold": 0.5, "delay-after-error": "30s", "min-plate-height": 0, "flag-save-failed": false, "flag-process-special": false, "char-iou-threshold": 0.7, "lpd-net-model-name": "lpdnet_yolo", "lpr-net-model-name": "lprnet_yolo", "vehicle-confidence": 0.7, "special-confidence": 0.9, "lpd-net-input-width": 640, "lpr-net-input-width": 160, "delay-between-frames": "1s", "lpd-net-input-height": 640, "lpr-net-input-height": 160, "vehicle-iou-threshold": 0.45, "max-capture-error-count": 3, "vd-net-model-name": "vdnet_yolo", "lpd-net-inference-server": "127.0.0.1:8000", "lpr-net-inference-server": "127.0.0.1:8000", "vd-net-input-width": 640, "lpd-net-input-tensor-name": "images", "lpr-net-input-tensor-name": "images", "lpr-net-output-tensor-name": "output0", "vd-net-input-height": 640, "lpd-net-output-tensor-name": "output0", "vehicle-area-ratio-threshold": 0.01, "vd-net-inference-server": "127.0.0.1:8000", "vd-net-input-tensor-name": "images", "vd-net-output-tensor-name": "output0", "vc-net-inference-server": "127.0.0.1:8000", "vc-net-model-name": "vcnet_vit", "vc-net-input-width": 224, "vc-net-input-height": 224, "vc-net-input-tensor-name": "input", "vc-net-output-tensor-name": "output"}') on conflict do nothing;
