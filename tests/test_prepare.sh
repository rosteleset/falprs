#!/bin/bash

# External variables used in the script

# PG_PORT - PostgreSQL port

pg_port="${pg_port:=5432}"

BASEDIR=$(realpath `dirname $0`)
sudo -u postgres psql < $BASEDIR/prepare_data.sql
PG_USER_FRS=test_falprs PG_PASSWD_FRS=123 PG_HOST_FRS=localhost PG_PORT_FRS=$pg_port PG_DB_FRS=test_frs $BASEDIR/../scripts/sql_frs.sh
PG_USER_LPRS=test_falprs PG_PASSWD_LPRS=123 PG_HOST_LPRS=localhost PG_PORT_LPRS=$pg_port PG_DB_LPRS=test_lprs $BASEDIR/../scripts/sql_lprs.sh
mkdir -p /tmp/test_falprs/static
cp --update=none $BASEDIR/images/* /tmp/test_falprs/static/

cd $BASEDIR/../build
cp --update=none $BASEDIR/test_config.yaml /tmp/test_falprs
sed -i 's/:5432/:'"$pg_port"'/' /tmp/test_falprs/test_config.yaml
./falprs -c /tmp/test_falprs/test_config.yaml & echo $! > /tmp/test_falprs/falprs.pid
