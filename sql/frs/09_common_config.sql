create table if not exists common_config
(
  id_group integer not null
    constraint common_config_pk
      primary key
    constraint common_config_vstream_groups_id_group_fk
      references vstream_groups
      on update cascade on delete cascade,
  config   jsonb
);
comment on table common_config is 'Common configuration parameters';
insert into common_config (id_group, config) values ((select id_group from vstream_groups where group_name = 'default'), '{"callback-timeout":"2s","comments-blurry-face":"The face image is not clear enough for registration.","comments-descriptor-creation-error":"Failed to register descriptor.","comments-descriptor-exists":"The descriptor already exists.","comments-inference-error":"Error: Triton Inference Server request failed.","comments-new-descriptor":"A new descriptor has been created.","comments-no-faces":"There are no faces in the image.","comments-non-frontal-face":"The face in the image must be frontal.","comments-non-normal-face-class":"A person wearing a mask or dark glasses.","comments-partial-face":"The face must be fully visible in the image.","comments-url-image-error":"Failed to receive image.","dnn-bd-input-height":320,"dnn-bd-input-tensor-name":"images","dnn-bd-input-width":320,"dnn-bd-model-name":"barcode_detection","dnn-bd-output-tensor-name":"output0","dnn-fc-input-height":192,"dnn-fc-input-tensor-name":"input.1","dnn-fc-input-width":192,"dnn-fc-model-name":"genet","dnn-fc-output-size":3,"dnn-fc-output-tensor-name":"419","dnn-fd-input-height":320,"dnn-fd-input-tensor-name":"input.1","dnn-fd-input-width":320,"dnn-fd-model-name":"scrfd","dnn-fr-input-height":112,"dnn-fr-input-tensor-name":"input.1","dnn-fr-input-width":112,"dnn-fr-model-name":"arcface","dnn-fr-output-size":512,"dnn-fr-output-tensor-name":"683","flag-copy-event-data":false,"inference-timeout":"1s","sg-max-descriptor-count":1000}') on conflict do nothing;
