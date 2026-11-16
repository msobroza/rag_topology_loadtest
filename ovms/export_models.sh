#!/usr/bin/env bash
# Export the reranker and the embedder to OpenVINO IR in an OVMS model repository,
# using the export_model.py helper shipped with the pinned OVMS tag.
#
# The weight format (fp32 or int8) is RAG_OPENVINO_WEIGHT_FORMAT, the setting the
# in-process export reads too, printed by rag-served-models. A models folder holds
# one export, of one weight format, recorded in MODELS_DIR/weight-format once both
# configs are written. A folder holding a config but no such file was exported
# before the format was recorded, always at int8. The script refuses to export
# into a folder that holds another weight format.
#
# Whether the embedder returns unit-length vectors is RAG_EMBEDDING_NORMALIZE, the
# setting the in-process embedder reads, printed by rag-served-models as
# EMBEDDING_NORMALIZE. The exporter normalises unless told --skip_normalize, and
# rewrites the embedder's graph on every run, so an export served as it is agrees.
#
# Usage: ovms/export_models.sh [MODELS_DIR]      (default: ovms/models)
# Needs: HF Hub access to download the source models; a Python with pip; this
# package installed (`make install`) so `rag-served-models` is on PATH.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=versions.env
source "$HERE/versions.env"

# The served models are pinned by the RagSettings defaults, not here. Asked of the
# command, not "$PYTHON -m", because $PYTHON may be a separate exporter virtualenv.
if ! command -v rag-served-models >/dev/null 2>&1; then
  echo "export_models.sh: rag-served-models not found on PATH; install this package (make install) so the command is available" >&2
  exit 1
fi
# Assigned first so a failing command stops the script (eval "$(...)" would hide it).
SERVED_MODELS=$(rag-served-models)
eval "$SERVED_MODELS"
# A rag-served-models older than the setting prints no weight format.
if [ -z "${OPENVINO_WEIGHT_FORMAT:-}" ]; then
  echo "export_models.sh: rag-served-models printed no OPENVINO_WEIGHT_FORMAT; reinstall this package (make install) so the command is current" >&2
  exit 1
fi
# A rag-served-models older than the embedder's normalisation prints none.
if [ -z "${EMBEDDING_NORMALIZE:-}" ]; then
  echo "export_models.sh: rag-served-models printed no EMBEDDING_NORMALIZE; reinstall this package (make install) so the command is current" >&2
  exit 1
fi
NORMALIZE_ARGS=()
if [ "$EMBEDDING_NORMALIZE" = false ]; then
  NORMALIZE_ARGS=(--skip_normalize)
fi

MODELS_DIR=${1:-$HERE/models}
WEIGHT_FORMAT_FILE="$MODELS_DIR/weight-format"
EXPORT_DIR="$MODELS_DIR/.export"
PYTHON=${PYTHON:-python}
# Pinned to the OVMS tag so the exporter matches the server that will load the IR.
EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/v${OVMS_VERSION}/demos/common/export_models"

# found_weight_format: the weight format of the export in $MODELS_DIR, "" if none.
found_weight_format() {
  if [ -f "$WEIGHT_FORMAT_FILE" ]; then
    cat "$WEIGHT_FORMAT_FILE"
  elif [ -f "$MODELS_DIR/config_reranker.json" ] || [ -f "$MODELS_DIR/config_embedder.json" ]; then
    echo int8  # exported before the format was recorded, when it was always int8
  fi
}

# OVMS's exporter skips a model folder that exists, so exporting another weight
# format here would leave the old weights under the new label.
FOUND_WEIGHT_FORMAT=$(found_weight_format)
if [ -n "$FOUND_WEIGHT_FORMAT" ] && [ "$FOUND_WEIGHT_FORMAT" != "$OPENVINO_WEIGHT_FORMAT" ]; then
  echo "export_models.sh: $MODELS_DIR already holds an export at weight format $FOUND_WEIGHT_FORMAT, and $OPENVINO_WEIGHT_FORMAT was asked (RAG_OPENVINO_WEIGHT_FORMAT); the exporter would keep the old weights; export into another folder: ovms/export_models.sh <another folder>" >&2
  exit 1
fi

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

fetch_if_missing export_model.py
fetch_if_missing requirements.txt

"$PYTHON" -m pip install -r "$EXPORT_DIR/requirements.txt"

"$PYTHON" "$EXPORT_DIR/export_model.py" rerank_ov \
  --source_model "$RERANKER_SOURCE_MODEL" \
  --model_name "$RERANKER_MODEL_NAME" \
  --weight-format "$OPENVINO_WEIGHT_FORMAT" \
  --config_file_path "$MODELS_DIR/config_reranker.json" \
  --model_repository_path "$MODELS_DIR"

"$PYTHON" "$EXPORT_DIR/export_model.py" embeddings_ov \
  --source_model "$EMBEDDER_SOURCE_MODEL" \
  --model_name "$EMBEDDER_MODEL_NAME" \
  --pooling CLS \
  --weight-format "$OPENVINO_WEIGHT_FORMAT" \
  --config_file_path "$MODELS_DIR/config_embedder.json" \
  --model_repository_path "$MODELS_DIR" \
  ${NORMALIZE_ARGS[@]+"${NORMALIZE_ARGS[@]}"}

# Written last: a folder records its weight format only once both configs exist.
echo "$OPENVINO_WEIGHT_FORMAT" > "$WEIGHT_FORMAT_FILE"

echo "export_models.sh: reranker config -> $MODELS_DIR/config_reranker.json"
echo "export_models.sh: embedder config -> $MODELS_DIR/config_embedder.json"
echo "export_models.sh: weight format $OPENVINO_WEIGHT_FORMAT -> $WEIGHT_FORMAT_FILE"
echo "export_models.sh: embedder normalize_embeddings $EMBEDDING_NORMALIZE -> $MODELS_DIR/$EMBEDDER_MODEL_NAME/graph.pbtxt"
