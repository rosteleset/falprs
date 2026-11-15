-- log_faces_id_log_index duplicated the primary key log_faces_pk (both on id_log): every insert into
-- log_faces maintained one more B-tree and wrote its WAL for nothing. Lookups by id_log use the
-- primary key. Dropped concurrently, so that a running service keeps inserting while it is dropped.
drop index concurrently if exists log_faces_id_log_index;
