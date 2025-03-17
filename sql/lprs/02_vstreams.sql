create table if not exists vstreams
(
    id_vstream serial
        constraint vstreams_pk
            primary key,
    id_group   integer
        constraint vstreams_id_group_fk
            references vstream_groups
            on update cascade on delete cascade,
    ext_id     varchar(32) not null,
    config     jsonb,
    constraint vstreams_group_ext_unique
        unique (id_group, ext_id)
);
comment on column vstreams.id_vstream is 'Video stream identifier (primary key)';
comment on column vstreams.id_group is 'Group''s identifier of the video stream (foreign key)';
comment on column vstreams.ext_id is 'External identifier of the video stream';
comment on column vstreams.config is 'Video stream configuration';
