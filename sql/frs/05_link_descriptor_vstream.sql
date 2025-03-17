create table if not exists link_descriptor_vstream
(
  id_descriptor integer
    constraint link_descriptor_vstream_face_descriptors_id_descriptor_fk
      references face_descriptors
      on update cascade on delete cascade,
  id_vstream    integer
    constraint link_descriptor_vstream_video_streams_id_vstream_fk
      references video_streams
      on update cascade on delete cascade,
  last_updated  timestamp with time zone default now() not null,
  flag_deleted  boolean                  default false not null,
  constraint link_descriptor_vstream_pk        unique (id_descriptor, id_vstream)
);
comment on table link_descriptor_vstream is 'Linking a descriptor to a video stream';
comment on column link_descriptor_vstream.id_descriptor is 'Descriptor identifier';
comment on column link_descriptor_vstream.id_vstream is 'Video stream identifier';
comment on column link_descriptor_vstream.last_updated is 'Last update time (for cache operation)';
comment on column link_descriptor_vstream.flag_deleted is 'Sign of record deletion (for cache operation)';

create index if not exists link_descriptor_vstream_id_vstream_flag_deleted_index
  on link_descriptor_vstream (id_vstream, flag_deleted)
  where (not flag_deleted);

create index if not exists link_descriptor_vstream_last_updated_index
  on link_descriptor_vstream (last_updated);
