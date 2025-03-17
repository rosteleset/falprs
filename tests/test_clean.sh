#!/bin/bash

BASEDIR=$(realpath `dirname $0`)
kill -s SIGTERM `cat /tmp/test_falprs/falprs.pid` && pidwait `cat /tmp/test_falprs/falprs.pid`
sudo -u postgres psql < $BASEDIR/clean_data.sql
rm -R -f /tmp/test_falprs
