#!/usr/bin/env bash
# Domino App / Agent entry point for every role of the rag_load_test topologies.
# Pick the role with RAG_ROLE (a Project or App environment variable):
#   monolith          (default) rag-serve with the embedder and the reranker in-process
#   workflow          rag-serve calling OVMS for the reranker (split-reranker), or for
#                     the reranker and the embedder (split-all, RAG_EMBEDDER_BACKEND=ovms)
#   ovms-reranker     OpenVINO Model Server serving config_reranker.json
#   ovms-embedder     OpenVINO Model Server serving config_embedder.json
#   fastapi-reranker / fastapi-embedder   the same models behind rag-model-server
#                     (same /v3 contract as OVMS) for the FastAPI-vs-OVMS experiment
# Domino only routes traffic to processes listening on 0.0.0.0:8888.
#
# OVMS has no base-path option. When Domino forwards the app path prefix
# (DOMINO_RUN_HOST_PATH) to the process instead of stripping it, set
# RAG_OVMS_PREFIX_PROXY=1: OVMS then listens on 127.0.0.1:9000 and nginx on
# 0.0.0.0:8888 strips the prefix (settings: domino/ovms-proxy.conf). Check first with
#   curl -i -H "Authorization: Bearer $(curl -s localhost:8899/access-token)" <App URL>/v3/models/<alias>
# (200 = prefix stripped, no proxy needed; 404 = enable the proxy).
set -euo pipefail

cd "$(dirname "$0")/.."

ROLE=${RAG_ROLE:-monolith}
MODELS_DIR=${RAG_OVMS_MODELS_DIR:-/mnt/data/ovms-models}
OVMS_UPSTREAM_PORT=9000

die() {
  echo "app.sh[$ROLE]: $*" >&2
  exit 1
}

serve_monolith() {
  export RAG_TOPOLOGY=${RAG_TOPOLOGY:-monolith}
  export RAG_EMBEDDER_BACKEND=local RAG_RERANKER_BACKEND=local
  exec rag-serve --host 0.0.0.0 --port 8888
}

serve_workflow() {
  [ -n "${RAG_OVMS_RERANK_URL:-}" ] \
    || die "set RAG_OVMS_RERANK_URL to the reranker App URL (the adapter appends /v3/rerank)"
  export RAG_RERANKER_BACKEND=ovms
  export RAG_EMBEDDER_BACKEND=${RAG_EMBEDDER_BACKEND:-local}
  local topology=split-reranker
  if [ "$RAG_EMBEDDER_BACKEND" = ovms ]; then
    [ -n "${RAG_OVMS_EMBEDDINGS_URL:-}" ] \
      || die "RAG_EMBEDDER_BACKEND=ovms needs RAG_OVMS_EMBEDDINGS_URL (embedder App URL; the adapter appends /v3/embeddings)"
    topology=split-all
  fi
  export RAG_TOPOLOGY=${RAG_TOPOLOGY:-$topology}
  # App-to-app calls inside Domino need the run's bearer token, fetched from
  # http://localhost:8899/access-token when RAG_OVMS_AUTH=domino.
  export RAG_OVMS_AUTH=${RAG_OVMS_AUTH:-domino}
  exec rag-serve --host 0.0.0.0 --port 8888
}

serve_fastapi_model() {
  exec rag-model-server --serve "$1" --host 0.0.0.0 --port 8888
}

# Render domino/ovms-proxy.conf: __PREFIX__ becomes DOMINO_RUN_HOST_PATH (the
# line is dropped when there is no prefix so "location /" is not duplicated).
render_proxy_config() {
  local prefix=${DOMINO_RUN_HOST_PATH:-}
  prefix=${prefix%/}
  if [ -n "$prefix" ]; then
    sed "s#__PREFIX__#$prefix#g" domino/ovms-proxy.conf > "$1"
  else
    sed '/__PREFIX__/d' domino/ovms-proxy.conf > "$1"
  fi
}

serve_ovms() {
  local config="$MODELS_DIR/$1"
  [ -f "$config" ] \
    || die "OVMS config not found: $config (fill a Domino Dataset with ovms/export_models.sh and mount it at $MODELS_DIR, or set RAG_OVMS_MODELS_DIR)"
  command -v ovms >/dev/null 2>&1 \
    || die "'ovms' binary not on PATH; use the environment built from domino/Dockerfile --target ovms"
  if [ "${RAG_OVMS_PREFIX_PROXY:-}" != 1 ]; then
    exec ovms --rest_port 8888 --config_path "$config"
  fi
  command -v nginx >/dev/null 2>&1 \
    || die "RAG_OVMS_PREFIX_PROXY=1 needs nginx (installed by domino/Dockerfile --target ovms)"
  local conf=/tmp/ovms-proxy.conf
  render_proxy_config "$conf"
  ovms --rest_bind_address 127.0.0.1 --rest_port "$OVMS_UPSTREAM_PORT" --config_path "$config" &
  local ovms_pid=$!
  nginx -c "$conf" -g 'daemon off;' &
  local nginx_pid=$!
  trap 'kill "$ovms_pid" "$nginx_pid" 2>/dev/null || true' EXIT
  # Exit when either process dies so Domino restarts the app (portable "wait -n").
  while kill -0 "$ovms_pid" 2>/dev/null && kill -0 "$nginx_pid" 2>/dev/null; do
    sleep 1
  done
  die "ovms or nginx exited"
}

case "$ROLE" in
  monolith) serve_monolith ;;
  workflow) serve_workflow ;;
  ovms-reranker) serve_ovms config_reranker.json ;;
  ovms-embedder) serve_ovms config_embedder.json ;;
  fastapi-reranker) serve_fastapi_model reranker ;;
  fastapi-embedder) serve_fastapi_model embedder ;;
  *) die "unknown RAG_ROLE '$ROLE'; expected monolith, workflow, ovms-reranker, ovms-embedder, fastapi-reranker or fastapi-embedder" ;;
esac
