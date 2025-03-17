create table if not exists face_descriptors
(
  id_descriptor   serial
    constraint face_descriptors_pk
      primary key,
  descriptor_data bytea,
  date_start      timestamp with time zone default now(),
  date_last       timestamp with time zone default now(),
  last_updated    timestamp with time zone default now() not null,
  flag_deleted    boolean                  default false not null,
  id_group        integer
    constraint face_descriptors_vstream_groups_id_group_fk
      references vstream_groups
      on update cascade on delete cascade
);
comment on table face_descriptors is 'Biometric parameters (descriptors) of faces';
comment on column face_descriptors.id_descriptor is 'Descriptor identifier';
comment on column face_descriptors.descriptor_data is 'Face descriptor (vector)';
comment on column face_descriptors.date_start is 'Descriptor start date';
comment on column face_descriptors.date_last is 'Date the descriptor was last used';
comment on column face_descriptors.last_updated is 'Last update time (for cache operation)';
comment on column face_descriptors.flag_deleted is 'Sign of record deletion (for cache operation)';
comment on column face_descriptors.id_group is 'Group identifier';

create index if not exists face_descriptors_id_descriptor_flag_deleted_index
  on face_descriptors (id_descriptor, flag_deleted)
  where (not flag_deleted);

create index if not exists face_descriptors_last_updated_index
  on face_descriptors (last_updated);
