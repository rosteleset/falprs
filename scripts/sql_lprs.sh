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
# PG_USER_LPRS - PostgreSQL user
# PG_PASSWD_LPRS - PostgreSQL password
# PG_HOST_LPRS - PostgreSQL server, or a comma-separated list of the servers of a cluster
# PG_PORT_LPRS - PostgreSQL port
# PG_DB_LPRS - PostgreSQL database

export PG_USER_LPRS="${PG_USER_LPRS:-falprs}"
export PG_PASSWD_LPRS="${PG_PASSWD_LPRS:-123}"
export PG_HOST_LPRS="${PG_HOST_LPRS:-localhost}"
export PG_PORT_LPRS="${PG_PORT_LPRS:-5432}"
export PG_DB_LPRS="${PG_DB_LPRS:-lprs}"

# Every host gets the port; with several hosts libpq connects to the one that accepts writes (the master)
PG_HOSTS_LPRS=$(echo "$PG_HOST_LPRS" | sed "s/,/:$PG_PORT_LPRS,/g; s/$/:$PG_PORT_LPRS/")
PG_URI_LPRS="postgresql://$PG_USER_LPRS:$PG_PASSWD_LPRS@$PG_HOSTS_LPRS/$PG_DB_LPRS?target_session_attrs=read-write"

psql -v ON_ERROR_STOP=1 "$PG_URI_LPRS" < "$BASEDIR/../sql/lprs/01_vstream_groups.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_LPRS" < "$BASEDIR/../sql/lprs/02_vstreams.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_LPRS" < "$BASEDIR/../sql/lprs/03_events_log.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_LPRS" < "$BASEDIR/../sql/lprs/04_default_vstream_config.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_LPRS" < "$BASEDIR/../sql/lprs/05_vstreams_new_columns.sql"
psql -v ON_ERROR_STOP=1 "$PG_URI_LPRS" < "$BASEDIR/../sql/lprs/06_triton_grpc_port.sql"
