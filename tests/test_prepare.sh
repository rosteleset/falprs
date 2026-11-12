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
# FALPRS_BIN - path to the service binary under test; required for cpp, whose sources are not in this branch
#              (FALPRS_IMPL must still name the implementation of the binary)

pg_port="${pg_port:=5432}"

FALPRS_IMPL="${FALPRS_IMPL:-rust}"
case "$FALPRS_IMPL" in
    rust)
        default_bin="$BASEDIR/../rust/target/release/falprs"
        ;;
    cpp)
        if [ -z "$FALPRS_BIN" ]; then
            echo "Error: FALPRS_IMPL=cpp needs FALPRS_BIN, the C++ binary built from the main branch"
            exit 1
        fi
        default_bin=""
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
# A second FRS stream group, for checks that a stream group cannot touch another one's objects
sudo -u postgres psql -d test_frs -c "insert into vstream_groups(group_name, auth_token) values ('test-group-2', '8a5c1e3e-5f4b-4c4e-9a43-2d6f0c1b7e21') on conflict do nothing"
mkdir -p /tmp/test_falprs/static
# The tests read the implementation under test from here: some API methods exist in one implementation only
echo "$FALPRS_IMPL" > /tmp/test_falprs/falprs.impl
cp --update=none $BASEDIR/images/* /tmp/test_falprs/static/

# The service writes test images to its current directory
cd "$(dirname "$falprs_bin")"
# The C++ reference reads the userver-shaped YAML config, the Rust service its TOML one
if [ "$FALPRS_IMPL" = cpp ]; then
    test_config=test_config.yaml
else
    test_config=test_config.toml
fi
cp --update=none $BASEDIR/$test_config /tmp/test_falprs
sed -i 's/:5432/:'"$pg_port"'/' /tmp/test_falprs/$test_config
"$falprs_bin" -c /tmp/test_falprs/$test_config & echo $! > /tmp/test_falprs/falprs.pid
