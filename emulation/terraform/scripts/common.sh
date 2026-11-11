#!/usr/bin/env bash
# Shared start of every emulation machine's startup script: Terraform puts this file
# (read with file(), not a template: plain bash) before the role's rendered
# template (elasticsearch.sh.tftpl, pool.sh.tftpl or workspace.sh.tftpl).
#
# A startup script runs as root on every boot, so every step is idempotent. Every
# network call has a timeout and bounded retries. No secret reaches the log (the
# serial console and journald): secrets go from a file to a file, never through a
# command line. Progress is published as the guest attribute rag-emulation/startup
# (running, then done or failed), readable from outside with
#   gcloud compute instances get-guest-attributes <machine> --zone <zone> --query-path=rag-emulation/startup
set -euo pipefail
umask 022

METADATA_URL=http://metadata.google.internal/computeMetadata/v1
export DEBIAN_FRONTEND=noninteractive

log() {
  echo "rag-emulation: $*" >&2
}

die() {
  log "$*"
  exit 1
}

# retry <attempts> <delay seconds> <command> [args...]: logs the command's name only.
retry() {
  local attempts=$1 delay=$2 n=1
  shift 2
  until "$@"; do
    [ "$n" -lt "$attempts" ] || { log "gave up after $n attempts: $1"; return 1; }
    log "attempt $n of $attempts failed: $1; retrying in ${delay}s"
    sleep "$delay"
    n=$((n + 1))
  done
}

guest_attribute() {
  curl -fsS --connect-timeout 5 --max-time 10 --retry 3 -X PUT --data "$2" \
    -H "Metadata-Flavor: Google" "$METADATA_URL/instance/guest-attributes/rag-emulation/$1"
}

on_exit() {
  local status=$?
  if [ "$status" -ne 0 ]; then guest_attribute startup failed || true; fi
}

# download <url> <file>: atomically, so an interrupted boot never leaves half a file.
download() {
  retry 5 15 curl -fsSL --connect-timeout 10 --max-time 1200 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

apt_install() {
  local missing=() package
  for package in "$@"; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed" \
      || missing+=("$package")
  done
  [ "${#missing[@]}" -gt 0 ] || return 0
  local options=(-q -o DPkg::Lock::Timeout=300 -o Acquire::http::Timeout=30 -o Acquire::Retries=3)
  retry 5 15 timeout 600 apt-get "${options[@]}" update
  retry 5 15 timeout 1800 apt-get "${options[@]}" install -y --no-install-recommends "${missing[@]}"
}

as_ubuntu() {
  sudo -u ubuntu -H -- "$@"
}

# The fake Domino logs in as ubuntu and runs Apps with sudo systemd-run.
ensure_ubuntu_user() {
  id ubuntu >/dev/null 2>&1 || useradd --create-home --shell /bin/bash ubuntu || id ubuntu >/dev/null
  printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' >/etc/sudoers.d/90-rag-emulation
  chmod 440 /etc/sudoers.d/90-rag-emulation
}

# install_uv <version>: the release binary into /usr/local/bin (on every PATH).
install_uv() {
  local version=$1 dir
  if [ "$(uv --version 2>/dev/null | awk '{print $2}')" = "$version" ]; then return 0; fi
  dir=$(mktemp -d)
  download "https://github.com/astral-sh/uv/releases/download/$version/uv-x86_64-unknown-linux-gnu.tar.gz" "$dir/uv.tar.gz"
  tar -xzf "$dir/uv.tar.gz" -C "$dir" --strip-components=1
  install -m 755 "$dir/uv" "$dir/uvx" /usr/local/bin/
  rm -rf "$dir"
}

# sync_repo <url> <ref> <path>: the clone at <path>, owned by ubuntu, checked out
# (detached) at <ref>, a branch, tag or commit. Untracked files such as deploy.yaml stay.
sync_repo() {
  local url=$1 ref=$2 path=$3
  if [ ! -d "$path/.git" ]; then
    install -d -m 755 "$(dirname "$path")"
    install -d -m 755 -o ubuntu -g ubuntu "$path"
    as_ubuntu git -C "$path" init -q
    as_ubuntu git -C "$path" remote add origin "$url"
  fi
  as_ubuntu git -C "$path" remote set-url origin "$url"
  retry 5 15 as_ubuntu timeout 600 git -C "$path" fetch --quiet --tags origin "$ref"
  as_ubuntu git -C "$path" checkout --quiet --force --detach FETCH_HEAD
}

# access_token_header <file>: the machine's service account token as a curl header
# file (mode 600), so the token is never on a command line.
access_token_header() {
  local token
  token=$(curl -fsS --connect-timeout 5 --max-time 10 --retry 5 -H "Metadata-Flavor: Google" \
    "$METADATA_URL/instance/service-accounts/default/token" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')
  printf 'Authorization: Bearer %s\n' "$token" >"$1"
}

# write_secret <project> <secret id> <file>: the latest version's payload into <file>,
# mode 600, owned by ubuntu. Retried for minutes: a new IAM grant takes time to apply.
write_secret() (
  umask 077
  local project=$1 secret=$2 out=$3 work
  work=$(mktemp -d)
  access_token_header "$work/header"
  retry 20 15 curl -fsS --connect-timeout 10 --max-time 30 -H "@$work/header" -o "$work/response.json" \
    "https://secretmanager.googleapis.com/v1/projects/$project/secrets/$secret/versions/latest:access"
  python3 -c 'import base64, json, sys; sys.stdout.buffer.write(base64.b64decode(json.load(open(sys.argv[1]))["payload"]["data"]))' \
    "$work/response.json" >"$work/payload"
  install -m 600 -o ubuntu -g ubuntu "$work/payload" "$out"
  rm -rf "$work"
)

# wait_for_http <url> <attempts>: 5 s apart.
wait_for_http() {
  retry "$2" 5 curl -fsS --connect-timeout 3 --max-time 10 -o /dev/null "$1"
}

# install_service <name> <unit file contents on stdin>: (re)started on every boot so
# a changed configuration is picked up.
install_service() {
  cat >"/etc/systemd/system/$1.service"
  systemctl daemon-reload
  systemctl enable "$1.service"
  systemctl restart "$1.service"
}

trap on_exit EXIT
guest_attribute startup running || true
