#!/bin/bash

set -euo pipefail

BASEDIR=$(realpath "$(dirname "$0")")

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source "$BASEDIR/.env"
elif [ -f "$BASEDIR/../.env" ]; then
    source "$BASEDIR/../.env"
fi

# External variables used in the script
# PG_USER_FRS - PostgreSQL user
# PG_PASSWD_FRS - PostgreSQL password
# PG_DB_FRS - PostgreSQL database
#
# PG_USER_LPRS - PostgreSQL user
# PG_PASSWD_LPRS - PostgreSQL password
# PG_DB_LPRS - PostgreSQL database

export PG_USER_FRS="${PG_USER_FRS:-falprs}"
export PG_PASSWD_FRS="${PG_PASSWD_FRS:-123}"
export PG_DB_FRS="${PG_DB_FRS:-frs}"

export PG_USER_LPRS="${PG_USER_LPRS:-falprs}"
export PG_PASSWD_LPRS="${PG_PASSWD_LPRS:-123}"
export PG_DB_LPRS="${PG_DB_LPRS:-lprs}"

sudo -u postgres psql \
    --set=frs_user="$PG_USER_FRS" \
    --set=frs_password="$PG_PASSWD_FRS" \
    --set=frs_db="$PG_DB_FRS" \
    --set=lprs_user="$PG_USER_LPRS" \
    --set=lprs_password="$PG_PASSWD_LPRS" \
    --set=lprs_db="$PG_DB_LPRS" \
    <<'SQL'
SELECT format(
    'CREATE USER %I WITH ENCRYPTED PASSWORD %L',
    :'frs_user',
    :'frs_password'
)
WHERE NOT EXISTS (
    SELECT FROM pg_catalog.pg_roles
    WHERE rolname = :'frs_user'
)\gexec

SELECT format(
    'CREATE USER %I WITH ENCRYPTED PASSWORD %L',
    :'lprs_user',
    :'lprs_password'
)
WHERE NOT EXISTS (
    SELECT FROM pg_catalog.pg_roles
    WHERE rolname = :'lprs_user'
)\gexec

SELECT format(
    'CREATE DATABASE %I OWNER %I',
    :'frs_db',
    :'frs_user'
)
WHERE NOT EXISTS (
    SELECT FROM pg_database
    WHERE datname = :'frs_db'
)\gexec

SELECT format(
    'CREATE DATABASE %I OWNER %I',
    :'lprs_db',
    :'lprs_user'
)
WHERE NOT EXISTS (
    SELECT FROM pg_database
    WHERE datname = :'lprs_db'
)\gexec

SELECT format(
    'GRANT ALL ON DATABASE %I TO %I',
    :'frs_db',
    :'frs_user'
)\gexec

SELECT format(
    'GRANT ALL ON DATABASE %I TO %I',
    :'lprs_db',
    :'lprs_user'
)\gexec
SQL
