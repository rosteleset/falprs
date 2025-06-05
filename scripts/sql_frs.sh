#!/bin/bash

# External variables used in the script

# PG_USER - PostgreSQL user
# PG_PASSWD - PostgreSQL password
# PG_HOST - PostgreSQL server
# PG_PORT - PostgreSQL port
# PG_DB - PostgreSQL database

BASEDIR=$(realpath `dirname $0`)

psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/01_vstream_groups.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/02_video_streams.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/03_face_descriptors.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/04_descriptor_images.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/05_link_descriptor_vstream.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/06_log_faces.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/07_special_groups.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/08_link_descriptor_sgroup.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/09_common_config.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/10_default_vstream_config.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/frs/11_face_descriptors_new_column.sql
