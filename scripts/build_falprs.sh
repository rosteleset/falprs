#!/bin/bash

# Build script for the FALPRS project
# Creates executable file and prepares a working directory

set -e

BASEDIR=$(realpath "$(dirname "$0")")

# Determine the user that invoked sudo.
# Build artifacts must belong to the regular user, not root.
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
    BUILD_USER="$SUDO_USER"
    BUILD_HOME="$(getent passwd "$BUILD_USER" | cut -d: -f6)"
else
    BUILD_USER="$(id -un)"
    BUILD_HOME="$HOME"
fi

# Load configuration from file if exists
if [ -f "$BASEDIR/.env" ]; then
    source "$BASEDIR/.env"
elif [ -f "$BASEDIR/../.env" ]; then
    source "$BASEDIR/../.env"
fi

# External variables used in the script
# FALPRS_WORKDIR - FALPRS working directory

export FALPRS_WORKDIR="${FALPRS_WORKDIR:-/opt/falprs}"
FALPRS_BINARY="$BASEDIR/../rust/target/release/falprs"
echo "Building FALPRS"

# The Rust build needs a C compiler only for the TLS crypto library; Rust itself is installed with rustup
APT_PACKAGES="build-essential ca-certificates curl python3-yaml"

# These operations require root privileges.
apt-get update
apt-get install -y $APT_PACKAGES

# Build FALPRS as the regular user.
# runuser does not request another password when the script is already running as root.
runuser -u "$BUILD_USER" -- env \
    HOME="$BUILD_HOME" \
    USER="$BUILD_USER" \
    LOGNAME="$BUILD_USER" \
    BASEDIR="$BASEDIR" \
    bash -c '
set -e

export PATH="$HOME/.cargo/bin:$PATH"

# rustup in the home directory of the building user; the shell profile is left as is
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
fi

cd "$BASEDIR/../rust"

# Install, or update, the toolchain channel named in rust-toolchain.toml
TOOLCHAIN=$(sed -n "s/^channel *= *\"\(.*\)\"/\1/p" rust-toolchain.toml)
rustup toolchain install "$TOOLCHAIN" --profile minimal

cargo build --release --locked -p falprs
'

# Everything below this point is deployment/configuration and requires root.
# The executable was built by BUILD_USER and is copied to the root-owned
# FALPRS working directory.

mkdir -p "$FALPRS_WORKDIR"
mkdir -p "$FALPRS_WORKDIR/static"

if cp "$FALPRS_BINARY" "$FALPRS_WORKDIR/falprs"; then
    # The update script rebuilds a working directory that holds another implementation
    echo rust > "$FALPRS_WORKDIR/falprs.impl"
else
    echo "Warning: failed to copy falprs, continuing..."
fi

cd "$BASEDIR/.."

cp --update=none ./examples/lprs/test001.jpg "$FALPRS_WORKDIR/static/"
cp --update=none ./examples/frs/einstein_001.jpg "$FALPRS_WORKDIR/static/"
cp --update=none ./examples/frs/einstein_002.jpg "$FALPRS_WORKDIR/static/"

python3 "$BASEDIR/merge_yaml.py" config.yaml.example "$FALPRS_WORKDIR/config.yaml"

CPU=$(nproc)

awk -v u_lprs="$PG_USER_LPRS" -v p_lprs="$PG_PASSWD_LPRS" -v h_lprs="$PG_HOST_LPRS" -v pt_lprs="$PG_PORT_LPRS" -v d_lprs="$PG_DB_LPRS" \
    -v u_frs="$PG_USER_FRS" -v p_frs="$PG_PASSWD_FRS" -v h_frs="$PG_HOST_FRS" -v pt_frs="$PG_PORT_FRS" -v d_frs="$PG_DB_FRS" \
    -v cpu="$CPU" '
function build_conn(u,p,h,pt,d) {
    return "postgresql://" u ":" p "@" h ":" pt "/" d
}

/^[[:space:]]*main-task-processor:/ { section="main" }
/^[[:space:]]*fs-task-processor:/   { section="fs" }
section=="main" && /^[[:space:]]*worker_threads:/ {
    sub(/[0-9]+/, cpu)
}
section=="fs" && /^[[:space:]]*worker_threads:/ {
    sub(/[0-9]+/, cpu*4)
}

$0 ~ /^[[:space:]]*lprs-postgresql-database:/ { section="lprs" }
$0 ~ /^[[:space:]]*frs-postgresql-database:/  { section="frs"  }

/^[[:space:]]*lprs-postgresql-database:/ || /^[[:space:]]*frs-postgresql-database:/ {
    print
    next
}

section=="lprs" && /^[[:space:]]*dbconnection:/ {
    sub(/dbconnection:.*/, "dbconnection: \x27" build_conn(u_lprs,p_lprs,h_lprs,pt_lprs,d_lprs) "\x27")
    print
    next
}

section=="frs" && /^[[:space:]]*dbconnection:/ {
    sub(/dbconnection:.*/, "dbconnection: \x27" build_conn(u_frs,p_frs,h_frs,pt_frs,d_frs) "\x27")
    print
    next
}

/^[^[:space:]]/ { section="" }

{ print }
' "$FALPRS_WORKDIR/config.yaml" > /tmp/config.yaml.new && mv /tmp/config.yaml.new "$FALPRS_WORKDIR/config.yaml"
