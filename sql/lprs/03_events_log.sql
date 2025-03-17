create table if not exists events_log
(
    id_event   serial
        constraint events_log_pk
            primary key,
    id_vstream integer                                not null
        constraint events_log_id_vstream_fk
            references vstreams
            on update cascade on delete cascade,
    log_date   timestamp with time zone default now() not null,
    info       jsonb,
    constraint events_log_unique
        unique (id_vstream, log_date)
);
comment on column events_log.id_event is 'Event record identifier (primary key)';
comment on column events_log.id_vstream is 'Video stream identifier';
comment on column events_log.log_date is 'Event''s date and time';
comment on column events_log.info is 'Event''s additional information';
