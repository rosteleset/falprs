#!/bin/bash

# Prepares the service config in the working directory:
# - an existing config.toml is left as is;
# - an old config.yaml is converted to config.toml once and kept as config.yaml.bak;
# - otherwise config.toml is created from config.toml.example with the database DSNs from .env.

set -e

BASEDIR=$(realpath "$(dirname "$0")")

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source "$BASEDIR/.env"
elif [ -f "$BASEDIR/../.env" ]; then
    source "$BASEDIR/../.env"
fi

# External variables used in the script
# FALPRS_WORKDIR - FALPRS working directory
# PG_USER_LPRS, PG_PASSWD_LPRS, PG_HOST_LPRS, PG_PORT_LPRS, PG_DB_LPRS - LPRS database connection
# PG_USER_FRS, PG_PASSWD_FRS, PG_HOST_FRS, PG_PORT_FRS, PG_DB_FRS - FRS database connection

FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"
# The same defaults as in sql_lprs.sh and sql_frs.sh
PG_USER_LPRS="${PG_USER_LPRS:-falprs}"
PG_PASSWD_LPRS="${PG_PASSWD_LPRS:-123}"
PG_HOST_LPRS="${PG_HOST_LPRS:-localhost}"
PG_PORT_LPRS="${PG_PORT_LPRS:-5432}"
PG_DB_LPRS="${PG_DB_LPRS:-lprs}"
PG_USER_FRS="${PG_USER_FRS:-falprs}"
PG_PASSWD_FRS="${PG_PASSWD_FRS:-123}"
PG_HOST_FRS="${PG_HOST_FRS:-localhost}"
PG_PORT_FRS="${PG_PORT_FRS:-5432}"
PG_DB_FRS="${PG_DB_FRS:-frs}"
CONFIG="$FALPRS_WORKDIR/config.toml"
OLD_CONFIG="$FALPRS_WORKDIR/config.yaml"

if [ -f "$CONFIG" ]; then
    exit 0
fi

if [ -f "$OLD_CONFIG" ]; then
    echo "Converting $OLD_CONFIG to $CONFIG"
    python3 "$BASEDIR/convert_config.py" "$OLD_CONFIG" "$CONFIG"
    mv "$OLD_CONFIG" "$OLD_CONFIG.bak"
    echo "The old config is kept as $OLD_CONFIG.bak"
    exit 0
fi

echo "Creating $CONFIG"
awk -v lprs="postgresql://$PG_USER_LPRS:$PG_PASSWD_LPRS@$PG_HOST_LPRS:$PG_PORT_LPRS/$PG_DB_LPRS" \
    -v frs="postgresql://$PG_USER_FRS:$PG_PASSWD_FRS@$PG_HOST_FRS:$PG_PORT_FRS/$PG_DB_FRS" '
/^\[/ { section = $0 }
section == "[lprs.database]" && /^dsn[[:space:]]*=/ { $0 = "dsn = \"" lprs "\"" }
section == "[frs.database]" && /^dsn[[:space:]]*=/ { $0 = "dsn = \"" frs "\"" }
{ print }
' "$BASEDIR/../config.toml.example" > "$CONFIG.new"
mv "$CONFIG.new" "$CONFIG"
