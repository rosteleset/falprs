create table if not exists special_groups
(
  id_special_group     serial
    constraint special_groups_pk
      primary key,
  group_name           varchar(200)                           not null,
  sg_api_token         varchar(64)                            not null,
  callback_url         varchar(200),
  max_descriptor_count integer                  default 1000  not null,
  id_group             integer
    constraint special_groups_vstream_groups_id_group_fk
      references vstream_groups
      on update cascade on delete cascade,
  last_updated         timestamp with time zone default now() not null,
  flag_deleted         boolean                  default false not null,
  constraint special_groups_pk_group_name
    unique (id_group, group_name)
);
comment on table special_groups is 'Special groups with processing on all video streams';
comment on column special_groups.id_special_group is 'Special group identifier';
comment on column special_groups.group_name is 'Special group name';
comment on column special_groups.sg_api_token is 'Token for working with API methods of a special group';
comment on column special_groups.callback_url is 'URL to call when a face recognition event occurs';
comment on column special_groups.max_descriptor_count is 'Maximum number of descriptors in a special group';
comment on column special_groups.id_group is 'Group identifier';
comment on column special_groups.last_updated is 'Last update time (for cache operation)';
comment on column special_groups.flag_deleted is 'Sign of record deletion (for cache operation)';

create index if not exists special_groups_last_updated_index
  on special_groups (last_updated);

create index if not exists special_groups_id_vgroup_flag_deleted_index
  on special_groups (id_special_group, flag_deleted)
  where (not flag_deleted);
