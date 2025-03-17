#!/bin/bash

# External variables used in the script

# PG_PORT - PostgreSQL port

pg_port="${pg_port:=5432}"

BASEDIR=$(realpath `dirname $0`)
sudo -u postgres psql < $BASEDIR/prepare_data.sql
pg_user=test_falprs pg_passwd=123 pg_host=localhost pg_port=$pg_port pg_db=test_frs $BASEDIR/../scripts/sql_frs.sh
pg_user=test_falprs pg_passwd=123 pg_host=localhost pg_port=$pg_port pg_db=test_lprs $BASEDIR/../scripts/sql_lprs.sh
mkdir -p /tmp/test_falprs/static
cp -n $BASEDIR/images/* /tmp/test_falprs/static/

cd $BASEDIR/../build
cp -n $BASEDIR/test_config.yaml /tmp/test_falprs
sed -i 's/:5432/:'"$pg_port"'/' /tmp/test_falprs/test_config.yaml
./falprs -c /tmp/test_falprs/test_config.yaml & echo $! > /tmp/test_falprs/falprs.pid
