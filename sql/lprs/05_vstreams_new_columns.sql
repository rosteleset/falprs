-- Soft deletion of video streams, as in FRS: removeStream marks the row, and the service deletes it
-- (with its events) once flag-deleted-ttl has passed, so that a recognition pipeline still running for
-- a removed stream can record its event. Used by the Rust implementation; the C++ one deletes rows at
-- once and does not read these columns, so it would take a marked row for a live stream.
alter table vstreams
    add if not exists flag_deleted boolean default false not null;
alter table vstreams
    add if not exists last_updated timestamp with time zone default now() not null;
comment on column vstreams.flag_deleted is 'Sign of record deletion (for cache operation)';
comment on column vstreams.last_updated is 'Time of the last change of the record';
