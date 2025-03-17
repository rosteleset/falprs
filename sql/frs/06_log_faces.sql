create table if not exists log_faces
(
  id_log         serial
    constraint log_faces_pk
      primary key,
  id_vstream     integer                                            not null
    constraint log_faces_video_streams_id_vstream_fk
      references video_streams
      on update cascade on delete cascade,
  log_date       timestamp with time zone default now() not null,
  id_descriptor  integer
    constraint log_faces_face_descriptors_id_descriptor_fk
      references face_descriptors
      on update cascade on delete set null,
  quality        double precision,
  face_left      integer,
  face_top       integer,
  face_width     integer,
  face_height    integer,
  screenshot_url varchar(200),
  copy_data      smallint                 default 0                 not null,
  log_uuid       uuid,
  ext_event_uuid varchar(64)
);
comment on table log_faces is 'Face event log';
comment on column log_faces.id_log is 'Log entry identifier';
comment on column log_faces.id_vstream is 'Video stream identifier';
comment on column log_faces.log_date is 'Logging date and time';
comment on column log_faces.id_descriptor is 'Descriptor identifier';
comment on column log_faces.quality is 'Face quality in the screenshot';
comment on column log_faces.face_left is 'X coordinate of the rectangular face area';
comment on column log_faces.face_top is 'Y coordinate of the rectangular face area';
comment on column log_faces.face_width is 'Width of rectangular face area';
comment on column log_faces.face_height is 'Height of rectangular face area';
comment on column log_faces.screenshot_url is 'Screenshot URL';
comment on column log_faces.copy_data is 'Flag to be used in special groups:
0 - do not copy data
1 - scheduled copy
2 - done copy';
comment on column log_faces.log_uuid is 'uuid of the log';

create index if not exists log_faces_copy_data_index
  on log_faces (copy_data);

create index if not exists log_faces_id_log_index
  on log_faces (id_log);

create index if not exists log_faces_id_vstream_log_date_index
  on log_faces (id_vstream, log_date);
