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
FALPRS_BINARY="$BASEDIR/../target/release/falprs"
echo "Building FALPRS"

# The Rust build needs a C compiler only for the TLS crypto library; Rust itself is installed with rustup
# python3-yaml is needed to convert an old config.yaml
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

cd "$BASEDIR/.."

# The workspace was moved from rust/ to the repository root; the old build results take gigabytes
if [ -d rust/target ] && [ ! -e rust/Cargo.toml ]; then
    echo "Removing the old build results in rust/target"
    rm -rf rust/target
    rmdir rust 2>/dev/null || true
fi

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

"$BASEDIR/prepare_config.sh"
