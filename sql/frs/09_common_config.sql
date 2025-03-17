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
insert into common_config (id_group, config) values ((select id_group from vstream_groups where group_name = 'default'), '{"callback-timeout": "2s", "comments-no-faces": "There are no faces in the image.", "dnn-fc-model-name": "genet", "dnn-fd-model-name": "scrfd", "dnn-fr-model-name": "arcface", "dnn-fc-input-width": 192, "dnn-fc-output-size": 3, "dnn-fd-input-width": 320, "dnn-fr-input-width": 112, "dnn-fr-output-size": 512, "dnn-fc-input-height": 192, "dnn-fd-input-height": 320, "dnn-fr-input-height": 112, "comments-blurry-face": "The face image is not clear enough for registration.", "flag-copy-event-data": false, "comments-partial-face": "The face must be fully visible in the image.", "comments-new-descriptor": "A new descriptor has been created.", "sg-max-descriptor-count": 1000, "comments-inference-error": "Error: Triton Inference Server request failed.", "comments-url-image-error": "Failed to receive image.", "dnn-fc-input-tensor-name": "input.1", "dnn-fd-input-tensor-name": "input.1", "dnn-fr-input-tensor-name": "input.1", "comments-non-frontal-face": "The face in the image must be frontal.", "dnn-fc-output-tensor-name": "419", "dnn-fr-output-tensor-name": "683", "comments-descriptor-exists": "The descriptor already exists.", "comments-non-normal-face-class": "A person wearing a mask or dark glasses.", "comments-descriptor-creation-error": "Failed to register descriptor."}') on conflict do nothing;
