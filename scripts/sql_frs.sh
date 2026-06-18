#!/bin/bash

set -e

BASEDIR=$(realpath `dirname $0`)

# External variables used in the script
# PG_USER_FRS - PostgreSQL user
# PG_PASSWD_FRS - PostgreSQL password
# PG_HOST_FRS - PostgreSQL server
# PG_PORT_FRS - PostgreSQL port
# PG_DB_FRS - PostgreSQL database

export PG_USER_FRS="${PG_USER_FRS:-falprs}"
export PG_PASSWD_FRS="${PG_PASSWD_FRS:-123}"
export PG_HOST_FRS="${PG_HOST_FRS:-localhost}"
export PG_PORT_FRS="${PG_PORT_FRS:-5432}"
export PG_DB_FRS="${PG_DB_FRS:-frs}"

psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/01_vstream_groups.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/02_video_streams.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/03_face_descriptors.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/04_descriptor_images.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/05_link_descriptor_vstream.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/06_log_faces.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/07_special_groups.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/08_link_descriptor_sgroup.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/09_common_config.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/10_default_vstream_config.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/11_face_descriptors_new_column.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/12_video_streams_new_column.sql
psql postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS < $BASEDIR/../sql/frs/13_log_barcodes.sql
