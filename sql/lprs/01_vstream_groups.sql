create table if not exists vstream_groups
(
    id_group   serial
        constraint vstream_groups_pk
            primary key,
    group_name varchar(100) not null
        constraint group_name_unique
            unique,
    auth_token uuid         not null
        constraint auth_token_unique
            unique
);
comment on table vstream_groups is 'Video stream groups';
comment on column vstream_groups.id_group is 'Group identifier (primary key)';
comment on column vstream_groups.group_name is ' Group name (unique)';
comment on column vstream_groups.auth_token is 'Authentication token for API requests (unique)';
create extension if not exists "uuid-ossp";
insert into vstream_groups(group_name, auth_token) values('default', uuid_generate_v4()) on conflict do nothing;
