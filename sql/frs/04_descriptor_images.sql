create table if not exists descriptor_images
(
  id_descriptor integer not null
    constraint descriptor_images_pk
      primary key
    constraint descriptor_images_face_descriptors_id_descriptor_fk
      references face_descriptors
      on update cascade on delete cascade,
  mime_type     varchar(50),
  face_image    bytea
);
comment on table descriptor_images is 'Descriptor images';
comment on column descriptor_images.id_descriptor is 'Descriptor identifier';
comment on column descriptor_images.mime_type is 'Image type';
comment on column descriptor_images.face_image is 'Face image data';
