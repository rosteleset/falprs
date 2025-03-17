create table if not exists video_streams
(
  id_vstream    serial
    constraint video_streams_pk
      primary key,
  vstream_ext   varchar(100)                           not null,
  url           varchar(200),
  callback_url  varchar(200),
  id_group      integer
    constraint video_streams_vstream_groups_id_group_fk
      references vstream_groups
      on update cascade on delete cascade,
  config        jsonb,
  last_updated  timestamp with time zone default now() not null,
  flag_deleted  boolean                  default false not null,
  constraint vstreams_group_ext_unique
    unique (id_group, vstream_ext)
);
comment on table video_streams is 'Information about video streams';
comment on column video_streams.id_vstream is 'Video stream identifier';
comment on column video_streams.vstream_ext is 'External video stream identifier';
comment on column video_streams.url is 'Frame capture URL';
comment on column video_streams.callback_url is 'URL for callbacks';
comment on column video_streams.id_group is 'Group''s identifier of the video stream (foreign key)';
comment on column video_streams.config is 'Video stream configuration';
comment on column video_streams.last_updated is 'Last update time (for cache operation)';
comment on column video_streams.flag_deleted is 'Sign of record deletion (for cache operation)';

create index if not exists video_streams_last_updated_index
  on video_streams (last_updated);

create index if not exists video_streams_id_vstream_flag_deleted_index
  on video_streams (id_vstream, flag_deleted)
  where (not flag_deleted);
