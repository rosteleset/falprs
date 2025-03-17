#!/bin/bash

# External variables used in the script

# PG_USER - PostgreSQL user
# PG_PASSWD - PostgreSQL password
# PG_HOST - PostgreSQL server
# PG_PORT - PostgreSQL port
# PG_DB - PostgreSQL database

BASEDIR=$(realpath `dirname $0`)

psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/lprs/01_vstream_groups.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/lprs/02_vstreams.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/lprs/03_events_log.sql
psql postgresql://$pg_user:$pg_passwd@$pg_host:$pg_port/$pg_db < $BASEDIR/../sql/lprs/04_default_vstream_config.sql
