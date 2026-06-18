#!/bin/bash

# Creates FALPRS service

set -e

BASEDIR=$(realpath `dirname $0`)

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source $BASEDIR/.env
elif [ -f "$BASEDIR/../.env" ]; then
    source $BASEDIR/../.env
fi

# External variables used in the script
# FALPRS_WORKDIR - FALPRS working directory

export FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"

cp $BASEDIR/../falprs.service.example /etc/systemd/system/falprs.service
cp $BASEDIR/../logrotate.example /etc/logrotate.d/falprs
groupadd falprs
useradd -g falprs -s /bin/true -d /dev/null falprs
mkdir -p /var/log/falprs/
chown -R falprs:falprs /var/log/falprs/
chown -R falprs:falprs $FALPRS_WORKDIR
systemctl daemon-reload
systemctl enable falprs
