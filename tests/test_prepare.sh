#!/bin/bash

BASEDIR=$(realpath `dirname $0`)

# Load configuration from file if exists, like the build and update scripts
if [ -f "$BASEDIR/../scripts/.env" ]; then
    source "$BASEDIR/../scripts/.env"
elif [ -f "$BASEDIR/../.env" ]; then
    source "$BASEDIR/../.env"
fi

# External variables used in the script

# PG_PORT - PostgreSQL port
# FALPRS_IMPL - implementation under test: rust (default) or cpp
# FALPRS_BIN - path to the service binary under test (overrides FALPRS_IMPL)

pg_port="${pg_port:=5432}"

FALPRS_IMPL="${FALPRS_IMPL:-rust}"
case "$FALPRS_IMPL" in
    rust)
        default_bin="$BASEDIR/../rust/target/release/falprs"
        ;;
    cpp)
        default_bin="$BASEDIR/../build/falprs"
        ;;
    *)
        echo "Error: FALPRS_IMPL must be rust or cpp, not '$FALPRS_IMPL'"
        exit 1
        ;;
esac
falprs_bin=$(realpath -m "${FALPRS_BIN:-$default_bin}")
if [ ! -x "$falprs_bin" ]; then
    echo "Error: the service binary $falprs_bin is not found; build it first"
    exit 1
fi
echo "Testing $falprs_bin"

sudo -u postgres psql < $BASEDIR/prepare_data.sql
$BASEDIR/../scripts/sql_frs.sh $BASEDIR/.env.tests
$BASEDIR/../scripts/sql_lprs.sh $BASEDIR/.env.tests
mkdir -p /tmp/test_falprs/static
cp --update=none $BASEDIR/images/* /tmp/test_falprs/static/

# The service writes test images to its current directory
cd "$(dirname "$falprs_bin")"
cp --update=none $BASEDIR/test_config.yaml /tmp/test_falprs
sed -i 's/:5432/:'"$pg_port"'/' /tmp/test_falprs/test_config.yaml
"$falprs_bin" -c /tmp/test_falprs/test_config.yaml & echo $! > /tmp/test_falprs/falprs.pid
