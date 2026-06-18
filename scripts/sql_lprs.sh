#!/bin/bash

set -e

BASEDIR=$(realpath `dirname $0`)

# External variables used in the script
# PG_USER_LPRS - PostgreSQL user
# PG_PASSWD_LPRS - PostgreSQL password
# PG_HOST_LPRS - PostgreSQL server
# PG_PORT_LPRS - PostgreSQL port
# PG_DB_LPRS - PostgreSQL database

export PG_USER_LPRS="${PG_USER_LPRS:-falprs}"
export PG_PASSWD_LPRS="${PG_PASSWD_LPRS:-123}"
export PG_HOST_LPRS="${PG_HOST_LPRS:-localhost}"
export PG_PORT_LPRS="${PG_PORT_LPRS:-5432}"
export PG_DB_LPRS="${PG_DB_LPRS:-lprs}"

psql postgresql://$PG_USER_LPRS:$PG_PASSWD_LPRS@$PG_HOST_LPRS:$PG_PORT_LPRS/$PG_DB_LPRS < $BASEDIR/../sql/lprs/01_vstream_groups.sql
psql postgresql://$PG_USER_LPRS:$PG_PASSWD_LPRS@$PG_HOST_LPRS:$PG_PORT_LPRS/$PG_DB_LPRS < $BASEDIR/../sql/lprs/02_vstreams.sql
psql postgresql://$PG_USER_LPRS:$PG_PASSWD_LPRS@$PG_HOST_LPRS:$PG_PORT_LPRS/$PG_DB_LPRS < $BASEDIR/../sql/lprs/03_events_log.sql
psql postgresql://$PG_USER_LPRS:$PG_PASSWD_LPRS@$PG_HOST_LPRS:$PG_PORT_LPRS/$PG_DB_LPRS < $BASEDIR/../sql/lprs/04_default_vstream_config.sql
