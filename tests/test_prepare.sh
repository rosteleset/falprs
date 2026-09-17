#!/bin/bash

# External variables used in the script

# PG_PORT - PostgreSQL port

pg_port="${pg_port:=5432}"

BASEDIR=$(realpath `dirname $0`)
sudo -u postgres psql < $BASEDIR/prepare_data.sql
$BASEDIR/../scripts/sql_frs.sh $BASEDIR/.env.tests
$BASEDIR/../scripts/sql_lprs.sh $BASEDIR/.env.tests
mkdir -p /tmp/test_falprs/static
cp --update=none $BASEDIR/images/* /tmp/test_falprs/static/

cd $BASEDIR/../build
cp --update=none $BASEDIR/test_config.yaml /tmp/test_falprs
sed -i 's/:5432/:'"$pg_port"'/' /tmp/test_falprs/test_config.yaml
./falprs -c /tmp/test_falprs/test_config.yaml & echo $! > /tmp/test_falprs/falprs.pid
