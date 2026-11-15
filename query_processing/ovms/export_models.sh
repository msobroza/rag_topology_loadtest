#!/usr/bin/env bash
# Export each embedder and reranker of the workflow file to OpenVINO IR (int8) in
# an OVMS model repository, using the export_model.py helper shipped with the pinned
# OVMS tag. Each component gets config_<component>.json, the config an OVMS
# model-server App of that component serves (domino/model_servers/ovms.sh).
#
# Usage: ovms/export_models.sh [MODELS_DIR]   (default: ovms/models in the working
# directory, the use case's, as every output of a command)
# Needs: HF Hub access to download the source models; a Python with pip; this
# package installed (`make install`) so `rag-served-models` is on PATH, run where it
# finds the use case's workflow file (its directory, or RAG_WORKFLOW_FILE).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=versions.env
source "$HERE/versions.env"

# The served models are pinned by the workflow file, not here. Asked of the
# command, not "$PYTHON -m", because $PYTHON may be a separate exporter virtualenv.
if ! command -v rag-served-models >/dev/null 2>&1; then
  echo "export_models.sh: rag-served-models not found on PATH; install this package (make install) so the command is available" >&2
  exit 1
fi
# Assigned first so a failing command stops the script (a pipe would hide it).
SERVED_MODELS=$(rag-served-models)

MODELS_DIR=${1:-$PWD/ovms/models}
EXPORT_DIR="$MODELS_DIR/.export"
PYTHON=${PYTHON:-python}
# Pinned to the OVMS tag so the exporter matches the server that will load the IR.
EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/v${OVMS_VERSION}/demos/common/export_models"

mkdir -p "$EXPORT_DIR"

fetch_if_missing() {
  local name=$1
  if [ -f "$EXPORT_DIR/$name" ]; then
    echo "export_models.sh: using cached $EXPORT_DIR/$name"
    return
  fi
  echo "export_models.sh: downloading $EXPORT_BASE_URL/$name"
  curl -fsSL -o "$EXPORT_DIR/$name" "$EXPORT_BASE_URL/$name"
}

# export_model <component> <type> <source id> <alias>: one model, and its config.
export_model() {
  local config="$MODELS_DIR/config_$1.json"
  case "$2" in
    reranker)
      "$PYTHON" "$EXPORT_DIR/export_model.py" rerank_ov \
        --source_model "$3" \
        --model_name "$4" \
        --weight-format int8 \
        --config_file_path "$config" \
        --model_repository_path "$MODELS_DIR"
      ;;
    embedder)
      # CLS: the pooling the RAG's embedder was trained with.
      "$PYTHON" "$EXPORT_DIR/export_model.py" embeddings_ov \
        --source_model "$3" \
        --model_name "$4" \
        --pooling CLS \
        --weight-format int8 \
        --config_file_path "$config" \
        --model_repository_path "$MODELS_DIR"
      ;;
    *)
      echo "export_models.sh: component '$1' is of type '$2'; expected an embedder or a reranker" >&2
      exit 1
      ;;
  esac
  echo "export_models.sh: $1 config -> $config"
}

fetch_if_missing export_model.py
fetch_if_missing requirements.txt

"$PYTHON" -m pip install -r "$EXPORT_DIR/requirements.txt"

# One line per model, its words shell-quoted by rag-served-models: eval only
# splits them back into the four arguments. Read on fd 3, so that the exporter
# cannot read the lines still to come from its standard input.
while IFS= read -r served <&3; do
  [ -n "$served" ] || continue
  eval "set -- $served"
  export_model "$@"
done 3<<<"$SERVED_MODELS"
