#!/bin/bash

set -e

BASEDIR=$(realpath "$(dirname "$0")")

# Load configuration from file if exists.
# An env file can be specified as the first argument.
if [ -n "$1" ]; then
    if [ -f "$1" ]; then
        source "$1"
    else
        echo "Error: env file not found: $1" >&2
        exit 1
    fi
elif [ -f "$BASEDIR/.env" ]; then
    source "$BASEDIR/.env"
elif [ -f "$BASEDIR/../.env" ]; then
    source "$BASEDIR/../.env"
fi

# External variables used in the script
# PG_USER_FRS - PostgreSQL user
# PG_PASSWD_FRS - PostgreSQL password
# PG_HOST_FRS - PostgreSQL server, or a comma-separated list of the servers of a cluster
# PG_PORT_FRS - PostgreSQL port
# PG_DB_FRS - PostgreSQL database

export PG_USER_FRS="${PG_USER_FRS:-falprs}"
export PG_PASSWD_FRS="${PG_PASSWD_FRS:-123}"
export PG_HOST_FRS="${PG_HOST_FRS:-localhost}"
export PG_PORT_FRS="${PG_PORT_FRS:-5432}"
export PG_DB_FRS="${PG_DB_FRS:-frs}"

# Every host gets the port; with several hosts libpq connects to the one that accepts writes (the master)
PG_HOSTS_FRS=$(echo "$PG_HOST_FRS" | sed "s/,/:$PG_PORT_FRS,/g; s/$/:$PG_PORT_FRS/")
PG_URI_FRS="postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOSTS_FRS/$PG_DB_FRS?target_session_attrs=read-write"

psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/01_vstream_groups.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/02_video_streams.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/03_face_descriptors.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/04_descriptor_images.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/05_link_descriptor_vstream.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/06_log_faces.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/07_special_groups.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/08_link_descriptor_sgroup.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/09_common_config.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/10_default_vstream_config.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/11_face_descriptors_new_column.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/12_video_streams_new_column.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/13_log_barcodes.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_FRS" < "$BASEDIR/../sql/frs/14_triton_grpc_port.sql"
