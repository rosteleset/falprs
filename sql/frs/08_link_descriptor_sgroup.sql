create table if not exists link_descriptor_sgroup
(
  id_descriptor integer                                not null
    constraint link_descriptor_sgroup_face_descriptors_id_descriptor_fk
      references face_descriptors
      on update cascade on delete cascade,
  id_sgroup     integer                                not null
    constraint link_descriptor_sgroup_special_groups_id_special_group_fk
      references special_groups
      on update cascade on delete cascade,
  last_updated  timestamp with time zone default now() not null,
  flag_deleted  boolean                  default false not null,
  constraint link_descriptor_sgroup_pk        unique (id_descriptor, id_sgroup)
);
comment on table link_descriptor_sgroup is 'Linking a descriptor to a special group';
comment on column link_descriptor_sgroup.last_updated is 'Last update time (for cache operation)';
comment on column link_descriptor_sgroup.flag_deleted is 'Sign of record deletion (for cache operation)';

create index if not exists link_descriptor_sgroup_last_updated_index
  on link_descriptor_sgroup (last_updated);

create index if not exists link_descriptor_sgroup_id_sgroup_flag_deleted_index
  on link_descriptor_sgroup (id_sgroup, flag_deleted)
  where (not flag_deleted);
