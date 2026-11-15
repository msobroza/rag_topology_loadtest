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
#
# The scripts of emulation/machine/ source this file for its functions (the cache
# bucket's), after setting QP_EMULATION_SOURCED=1.
set -euo pipefail
umask 022

METADATA_URL=http://metadata.google.internal/computeMetadata/v1
STORAGE_URL=https://storage.googleapis.com
export DEBIAN_FRONTEND=noninteractive

# The cache bucket's fixed layout, shared by the restore at boot and
# emulation/machine/cache_bucket.sh: the dataset folder <folder> is cached as the
# object models/<folder>.tar. The two folder names are also what
# workspace.sh.tftpl's deploy.yaml points the Apps at.
DATASET_OPENVINO=openvino
DATASET_OVMS=ovms-models
CACHE_MODELS_PREFIX=models/
CACHE_OPENVINO_PREFIX="$CACHE_MODELS_PREFIX$DATASET_OPENVINO/"
CACHE_OVMS_OBJECT="$CACHE_MODELS_PREFIX$DATASET_OVMS.tar"
# What exported_model_dir names a model's folder: the model id's last segment.
MODEL_FOLDER_PATTERN='[A-Za-z0-9_][A-Za-z0-9._-]*'
# Written at boot by write_cache_config (under the emulation's config folder);
# the variable is for the tests of emulation/machine/.
CACHE_ENV_FILE=${QP_EMULATION_CACHE_ENV:-/etc/rag-emulation/cache.env}
# A curl status for "the bucket has no such object" (curl's own codes stop at 99).
BUCKET_NO_SUCH_OBJECT=100

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

# --- The cache bucket (storage.tf), through Cloud Storage's XML and JSON APIs ---

# bucket_curl <curl args...>: curl with the machine's service account token, from a
# header file so that it is never on a command line. Transient failures (timeouts,
# 5xx, 429) are retried; HTTP errors are the caller's to read (-w '%{http_code}').
bucket_curl() {
  local header status=0
  header=$(mktemp)
  access_token_header "$header"
  curl -sS --connect-timeout 10 --max-time 1800 --retry 3 -H "@$header" "$@" || status=$?
  rm -f "$header"
  return "$status"
}

# bucket_fetch <bucket> <object> <file>: 0 once <file> holds the object (written
# atomically), BUCKET_NO_SUCH_OBJECT on HTTP 404; any other answer ends the script.
bucket_fetch() {
  local bucket=$1 object=$2 out=$3 status
  status=$(bucket_curl -o "$out.part" -w '%{http_code}' "$STORAGE_URL/$bucket/$object") || true
  case $status in
    200) mv "$out.part" "$out" ;;
    404) rm -f "$out.part"; return "$BUCKET_NO_SUCH_OBJECT" ;;
    *) rm -f "$out.part"; die "fetching gs://$bucket/$object answered HTTP $status, expected 200 or 404" ;;
  esac
}

# bucket_store <file> <bucket> <object>: uploads <file> as the object, replacing it.
bucket_store() {
  local status
  status=$(bucket_curl -T "$1" -o /dev/null -w '%{http_code}' "$STORAGE_URL/$2/$3") || true
  [ "$status" = 200 ] || die "storing $1 as gs://$2/$3 answered HTTP $status, expected 200"
}

# bucket_list_page <bucket> <prefix> <file>: one try at the JSON API's listing.
bucket_list_page() {
  local status
  status=$(bucket_curl -o "$3" -w '%{http_code}' "$STORAGE_URL/storage/v1/b/$1/o?prefix=$2") || true
  [ "$status" != 200 ] || return 0
  log "listing gs://$1/$2 answered HTTP $status, expected 200: $(head -c 300 "$3" 2>/dev/null)"
  return 1
}

# bucket_list_details <bucket> <prefix> <attempts>: "<name> <size> <updated>"
# (tab-separated) for each object under <prefix>. At boot <attempts> is large: a
# new IAM grant takes minutes to apply (see write_secret).
bucket_list_details() {
  local answer status=0
  answer=$(mktemp)
  retry "$3" 15 bucket_list_page "$1" "$2" "$answer" || status=$?
  [ "$status" -ne 0 ] || python3 -c '
import json, sys
answer = json.load(open(sys.argv[1]))
if answer.get("nextPageToken"):
    sys.exit("the listing has more than one page (1,000 objects): not supported")
for item in answer.get("items", []):
    print(item["name"], item["size"], item["updated"], sep="\t")
' "$answer" || status=$?
  rm -f "$answer"
  [ "$status" -eq 0 ] || die "could not list gs://$1/$2"
}

# bucket_list <bucket> <prefix> <attempts>: the object names under <prefix>.
bucket_list() {
  bucket_list_details "$@" | cut -f1
}

# write_cache_config <bucket> <dataset folder>: what emulation/machine/ reads (no
# secret: mode 644), and the dataset folder, owned by ubuntu as the Apps run as ubuntu.
write_cache_config() {
  install -d -m 755 "$(dirname "$CACHE_ENV_FILE")"
  printf 'CACHE_BUCKET=%s\nDATASET_DIR=%s\n' "$1" "$2" >"$CACHE_ENV_FILE.part"
  chmod 644 "$CACHE_ENV_FILE.part"
  mv "$CACHE_ENV_FILE.part" "$CACHE_ENV_FILE"
  owned_folder "$2"
}

# cache_config_value <name>: one value of the file write_cache_config wrote (read
# with sed, never evaluated).
cache_config_value() {
  local value
  [ -r "$CACHE_ENV_FILE" ] \
    || die "$CACHE_ENV_FILE not found: expected on a machine of the emulation (its startup script writes it)"
  value=$(sed -n "s/^$1=//p" "$CACHE_ENV_FILE")
  [ -n "$value" ] || die "$CACHE_ENV_FILE has no $1=<value> line"
  printf '%s\n' "$value"
}

# owned_folder <folder>: created (with its parents) if missing, owned by ubuntu.
owned_folder() {
  [ ! -d "$1" ] || return 0
  mkdir -p "$1"
  chown ubuntu:ubuntu "$1"
}

# cached_model_folder <object>: the dataset subfolder an object of the cache
# restores into; ends the script naming an object of any other shape.
cached_model_folder() {
  local name=${1#"$CACHE_OPENVINO_PREFIX"}
  if [ "$1" = "$CACHE_OVMS_OBJECT" ]; then
    echo "$DATASET_OVMS"
  elif [ "$name" != "$1" ] && [[ $name =~ ^$MODEL_FOLDER_PATTERN\.tar$ ]]; then
    echo "$DATASET_OPENVINO/${name%.tar}"
  else
    die "unexpected object $1 in the cache: expected $CACHE_OPENVINO_PREFIX<model folder>.tar or $CACHE_OVMS_OBJECT"
  fi
}

# restore_cached_model <bucket> <object> <dataset folder>: the folder back in the
# dataset, unless it is already there (a reboot must not clobber it). Unpacked
# beside its target and moved into place, so a half-restored folder never shows.
restore_cached_model() {
  local bucket=$1 object=$2 folder target staging
  folder=$(cached_model_folder "$object") || exit 1
  target="$3/$folder"
  if [ -d "$target" ]; then log "model kept: $target exists, gs://$bucket/$object not restored"; return 0; fi
  owned_folder "$(dirname "$target")"
  staging=$(mktemp -d "$(dirname "$target")/.restore.XXXXXX")
  bucket_fetch "$bucket" "$object" "$staging/model.tar" || die "gs://$bucket/$object was listed, then not found"
  tar -xf "$staging/model.tar" -C "$staging"
  [ -d "$staging/$(basename "$folder")" ] || die "gs://$bucket/$object does not hold the folder $(basename "$folder")"
  chown -R ubuntu:ubuntu "$staging/$(basename "$folder")"
  mv "$staging/$(basename "$folder")" "$target"
  rm -rf "$staging"
  log "model restored: $target from gs://$bucket/$object"
}

# restore_cached_models <bucket> <prefix> <dataset folder>: every cached model
# under <prefix>. An empty bucket is the first sitting's normal state.
restore_cached_models() {
  local objects object
  objects=$(bucket_list "$1" "$2" 20) || exit 1
  [ -n "$objects" ] || { log "no cached model under gs://$1/$2"; return 0; }
  for object in $objects; do
    restore_cached_model "$1" "$object" "$3"
  done
}

# A startup script reports its progress. A script that sources this file for its
# functions (QP_EMULATION_SOURCED=1) must not: its exit, failed or not, would set
# the machine's startup state, which `make ready` reads.
if [ "${QP_EMULATION_SOURCED:-}" != 1 ]; then
  trap on_exit EXIT
  guest_attribute startup running || true
fi
