#!/usr/bin/env bash
# The emulation's cache bucket, from a machine of the emulation (emulation/README.md,
# "The cache bucket"). Run from the machine's clone of the repository:
#   cache_bucket.sh models-save            upload this machine's exported models
#                                          (an object the bucket has is kept unless FORCE=1)
#   cache_bucket.sh results-save <folder>  upload <folder> as results/<UTC time>.tar.gz
#   cache_bucket.sh status                 each model's and result archive's name, size and last change
# The bucket and the dataset folder come from the file the startup script wrote
# (/etc/rag-emulation/cache.env; RAG_EMULATION_CACHE_ENV names another, for tests).
set -euo pipefail

# common.sh is the startup scripts' prelude; with this set it only defines functions.
export RAG_EMULATION_SOURCED=1
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../terraform/scripts/common.sh
. "$HERE/../terraform/scripts/common.sh"

# Fixed by the cache's layout (README): where result archives go.
CACHE_RESULTS_PREFIX=results/
# The two configs export_models.sh writes last: without both, the export is unfinished.
OVMS_CONFIGS="config_reranker.json config_embedder.json"

CACHE_BUCKET=$(cache_config_value CACHE_BUCKET)
DATASET_DIR=$(cache_config_value DATASET_DIR)
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rag-cache.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

# save_folder <folder> <object> <objects the bucket has>: one line, saved or kept.
save_folder() {
  local folder=$1 object=$2
  if [ "${FORCE:-}" != 1 ] && printf '%s\n' "$3" | grep -qxF "$object"; then
    echo "kept: gs://$CACHE_BUCKET/$object (FORCE=1 replaces it)"
    return 0
  fi
  # No compression: model weights do not compress.
  tar -cf "$WORK_DIR/upload.tar" -C "$(dirname "$folder")" "$(basename "$folder")"
  bucket_store "$WORK_DIR/upload.tar" "$CACHE_BUCKET" "$object"
  rm -f "$WORK_DIR/upload.tar"
  echo "saved: gs://$CACHE_BUCKET/$object"
}

# ovms_export_is_complete <folder>: both configs are there.
ovms_export_is_complete() {
  local config
  for config in $OVMS_CONFIGS; do
    [ -f "$1/$config" ] || return 1
  done
}

save_openvino_models() {
  local folder object
  for folder in "$DATASET_DIR/$DATASET_OPENVINO"/*/; do
    [ -d "$folder" ] || continue
    folder=${folder%/}
    object="$CACHE_OPENVINO_PREFIX$(basename "$folder").tar"
    cached_model_folder "$object" >/dev/null || exit 1
    save_folder "$folder" "$object" "$1"
  done
}

save_ovms_models() {
  local folder="$DATASET_DIR/$DATASET_OVMS"
  [ -d "$folder" ] || return 0
  if ! ovms_export_is_complete "$folder"; then
    echo "not saved: $folder lacks one of $OVMS_CONFIGS (an unfinished export)"
    return 0
  fi
  save_folder "$folder" "$CACHE_OVMS_OBJECT" "$1"
}

save_models() {
  local existing
  existing=$(bucket_list "$CACHE_BUCKET" "$CACHE_MODELS_PREFIX" 1) || exit 1
  save_openvino_models "$existing"
  save_ovms_models "$existing"
}

# save_results <folder>: prints the archive's gs:// URL.
save_results() {
  local folder=${1%/} object
  [ -d "$folder" ] || die "results-save: '$folder' is not a folder; expected the results folder to archive"
  object="$CACHE_RESULTS_PREFIX$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  tar -czf "$WORK_DIR/results.tar.gz" -C "$(dirname "$folder")" "$(basename "$folder")"
  bucket_store "$WORK_DIR/results.tar.gz" "$CACHE_BUCKET" "$object"
  echo "gs://$CACHE_BUCKET/$object"
}

# show_status: the models and the result archives. Not elasticsearch/: a snapshot is
# many files of Elasticsearch's own, and index_snapshot.sh list names the snapshots.
show_status() {
  local prefix listing
  for prefix in "$CACHE_MODELS_PREFIX" "$CACHE_RESULTS_PREFIX"; do
    listing=$(bucket_list_details "$CACHE_BUCKET" "$prefix" 1) || exit 1
    echo "gs://$CACHE_BUCKET/$prefix"
    printf '%s\n' "${listing:-(empty)}"
  done
}

case ${1:-} in
  models-save) save_models ;;
  results-save) save_results "${2:?results-save needs the folder to archive}" ;;
  status) show_status ;;
  *) die "unknown command '${1:-}'; expected models-save, results-save <folder> or status" ;;
esac
