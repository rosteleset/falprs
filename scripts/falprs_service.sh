#!/bin/bash

# External variables used in the script

# FALPRS_WORKDIR - FALPRS working directory

FALPRS_WORKDIR="${FALPRS_WORKDIR:=/opt/falprs}"

BASEDIR=$(realpath `dirname $0`)
cp $BASEDIR/../falprs.service.example /etc/systemd/system/falprs.service
cp $BASEDIR/../logrotate.example /etc/logrotate.d/falprs
groupadd falprs
useradd -g falprs -s /bin/true -d /dev/null falprs
mkdir -p /var/log/falprs/
chown -R falprs:falprs /var/log/falprs/
chown -R falprs:falprs $FALPRS_WORKDIR
systemctl daemon-reload
systemctl enable falprs
systemctl start falprs
