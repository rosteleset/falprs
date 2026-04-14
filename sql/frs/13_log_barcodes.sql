create table if not exists log_barcodes
(
    id_log   serial
        constraint log_barcodes_pk
            primary key,
    id_vstream integer                                not null
        constraint log_barcodes_video_streams_id_vstream_fk
            references video_streams
            on update cascade on delete cascade,
    log_date   timestamp with time zone default now() not null,
    info       jsonb,
    constraint log_barcodes_unique
        unique (id_vstream, log_date)
);
comment on table log_barcodes is 'Barcode event log';
comment on column log_barcodes.id_log is 'Log entry identifier (primary key)';
comment on column log_barcodes.id_vstream is 'Video stream identifier';
comment on column log_barcodes.log_date is 'Logging date and time';
comment on column log_barcodes.info is 'Event''s additional information';
