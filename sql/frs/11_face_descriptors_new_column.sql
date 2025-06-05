alter table face_descriptors
    add if not exists id_parent integer references face_descriptors(id_descriptor) on update cascade on delete cascade;
create index if not exists face_descriptors_id_parent_index
    on face_descriptors (id_parent);
comment on column face_descriptors.id_parent is 'Parent identifier for spawned descriptor';
