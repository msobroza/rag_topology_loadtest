#!/usr/bin/env bash
# Export the reranker and the embedder to OpenVINO IR (int8) in an OVMS model
# repository, using the export_model.py helper shipped with the pinned OVMS tag.
#
# Usage: ovms/export_models.sh [MODELS_DIR]      (default: ovms/models)
# Needs: HF Hub access to download the source models; a Python with pip.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=versions.env
source "$HERE/versions.env"

MODELS_DIR=${1:-$HERE/models}
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

fetch_if_missing export_model.py
fetch_if_missing requirements.txt

"$PYTHON" -m pip install -r "$EXPORT_DIR/requirements.txt"

"$PYTHON" "$EXPORT_DIR/export_model.py" rerank_ov \
  --source_model "$RERANKER_SOURCE_MODEL" \
  --model_name "$RERANKER_MODEL_NAME" \
  --weight-format int8 \
  --config_file_path "$MODELS_DIR/config_reranker.json" \
  --model_repository_path "$MODELS_DIR"

"$PYTHON" "$EXPORT_DIR/export_model.py" embeddings_ov \
  --source_model "$EMBEDDER_SOURCE_MODEL" \
  --model_name "$EMBEDDER_MODEL_NAME" \
  --pooling CLS \
  --weight-format int8 \
  --config_file_path "$MODELS_DIR/config_embedder.json" \
  --model_repository_path "$MODELS_DIR"

echo "export_models.sh: reranker config -> $MODELS_DIR/config_reranker.json"
echo "export_models.sh: embedder config -> $MODELS_DIR/config_embedder.json"
