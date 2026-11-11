#!/usr/bin/env bash
# Domino App / Agent entry point for every role of the rag_load_test topologies.
# Pick the role with RAG_ROLE (a Project or App environment variable):
#   monolith          (default) the workflow app for RAG_TOPOLOGY=monolith (the default)
#   workflow          the workflow app for RAG_TOPOLOGY=split-reranker (the default) or
#                     split-all
#   ovms-reranker     OpenVINO Model Server serving config_reranker.json
#   ovms-embedder     OpenVINO Model Server serving config_embedder.json
#                     (both from RAG_OVMS_MODELS_DIR, /mnt/data/ovms-models by default)
#   fastapi-reranker / fastapi-embedder   the same models behind rag-model-server
#                     (same /v3 contract as OVMS) for the FastAPI-vs-OVMS experiment
# Domino only routes traffic to processes listening on 0.0.0.0:8888.
#
# RAG_BOOTSTRAP=1 (set by rag-deploy) first installs what the role needs, for a
# generic environment with only uv: the OVMS roles apt-get libxml2 and curl (plus
# nginx with RAG_OVMS_PREFIX_PROXY=1, via sudo unless root) and run ovms from
# RAG_OVMS_BUNDLE (a folder with bin/ovms and lib/, or its .tar.gz); every other
# role uv-installs this project into a virtualenv under $HOME, with the extras named
# by RAG_BOOTSTRAP_EXTRAS (e.g. "domino") when set. Unset, nothing is installed:
# the prebuilt `rag` / `ovms` images already have it all.
#
# Models are provisioned on the App's own disk the first time it starts without
# them, then reused. An OVMS role whose config is missing from RAG_OVMS_MODELS_DIR
# runs ovms/export_models.sh there (it exports both models; needs uv and the Hub):
# the exporter's pinned requirements go into a virtualenv of their own
# ($HOME/ovms-exporter-venv) so they cannot downgrade the project's packages, and
# the project is uv-installed into $HOME/rag-venv first when `rag-served-models`
# is not on PATH (the OVMS roles' bootstrap installs no project). The in-process
# models (monolith, workflow, fastapi-*) export themselves into
# RAG_OPENVINO_MODELS_DIR on their first start (RAG_LOCAL_MODEL_RUNTIME=openvino).
#
# Both workflow app roles run `rag-serve --domino`: Python resolves the embedder and
# reranker backends from RAG_TOPOLOGY (src/rag_load_test/domain/topology.py), requires
# RAG_OVMS_*_URL for every port on a model server and defaults RAG_OVMS_AUTH=domino.
# A role here only picks its default topology and refuses one it does not serve.
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

# serve_workflow_app <default topology> <allowed topology>...
serve_workflow_app() {
  local default_topology=$1
  shift
  export RAG_TOPOLOGY=${RAG_TOPOLOGY:-$default_topology}
  case " $* " in
    *" $RAG_TOPOLOGY "*) ;;
    *) die "RAG_TOPOLOGY='$RAG_TOPOLOGY' is not served by this role; allowed: $*" ;;
  esac
  exec rag-serve --host 0.0.0.0 --port 8888 --domino
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
  if [ ! -f "$config" ]; then provision_ovms_models "$config"; fi
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

# provision_ovms_models <config>: export both models into $MODELS_DIR, once.
# Each App has its own disk, so no other process writes this folder: no lock.
provision_ovms_models() {
  local config=$1 exporter_venv="$HOME/ovms-exporter-venv"
  command -v uv >/dev/null 2>&1 \
    || die "OVMS config not found: $config, and 'uv' is not on PATH to export it with ovms/export_models.sh; install uv, or mount a folder holding it and set RAG_OVMS_MODELS_DIR"
  echo "app.sh[$ROLE]: $config not found; exporting the models into $MODELS_DIR (RAG_OVMS_MODELS_DIR) with ovms/export_models.sh" >&2
  # export_models.sh asks the project's rag-served-models which models to export.
  if ! command -v rag-served-models >/dev/null 2>&1; then
    install_project "$HOME/rag-venv" .
  fi
  # Its pinned requirements are pip-installed into $PYTHON, hence a virtualenv of
  # their own; --seed because the script runs "$PYTHON -m pip".
  if [ ! -x "$exporter_venv/bin/python" ]; then
    uv venv "$exporter_venv" --seed --python "$(cat .python-version)"
  fi
  PYTHON="$exporter_venv/bin/python" ovms/export_models.sh "$MODELS_DIR" \
    || die "exporting the OVMS models into $MODELS_DIR failed (ran ovms/export_models.sh with PYTHON=$exporter_venv/bin/python; it needs the Hugging Face Hub and the OVMS exporter's requirements); $config is missing; fix the error above, or mount a folder holding it and set RAG_OVMS_MODELS_DIR"
  [ -f "$config" ] \
    || die "ovms/export_models.sh finished but $config is still missing; check RAG_OVMS_MODELS_DIR=$MODELS_DIR"
}

# install_project <venv> <project>: uv-install this project, then put it on PATH.
install_project() {
  # Its own virtualenv, on the project's Python (.python-version), which uv
  # fetches when the environment lacks it: the environment's own packages
  # (an old urllib3 in Domino's standard image) break ours.
  uv venv "$1" --python "$(cat .python-version)"
  uv pip install --python "$1/bin/python" "$2"
  export PATH="$1/bin:$PATH"
}

# rag-deploy publishes into a generic environment with no image of our own, so
# the role's dependencies are installed when the App starts.
bootstrap() {
  case "$ROLE" in
    ovms-*) bootstrap_ovms ;;
    *)
      local project=.
      # An Agent's launch script sets RAG_BOOTSTRAP_EXTRAS=domino (Domino tracing).
      if [ -n "${RAG_BOOTSTRAP_EXTRAS:-}" ]; then project=".[$RAG_BOOTSTRAP_EXTRAS]"; fi
      install_project "$HOME/rag-venv" "$project"
      ;;
  esac
}

bootstrap_ovms() {
  local bundle=${RAG_OVMS_BUNDLE:-} packages="libxml2 curl" sudo=""
  [ -n "$bundle" ] && [ -e "$bundle" ] \
    || die "RAG_OVMS_BUNDLE='$bundle' not found; set it to the OVMS bundle (a folder with bin/ovms and lib/, or its .tar.gz)"
  if [ "${RAG_OVMS_PREFIX_PROXY:-}" = 1 ]; then packages="$packages nginx"; fi
  [ "$(id -u)" = 0 ] || sudo=sudo
  $sudo apt-get update
  # shellcheck disable=SC2086  # one word per package
  $sudo apt-get install -y --no-install-recommends $packages
  if [ -f "$bundle" ]; then
    local unpacked
    unpacked=$(mktemp -d)
    tar -xzf "$bundle" -C "$unpacked"
    bundle="$unpacked/ovms"
  fi
  export PATH="$bundle/bin:$PATH"
  export LD_LIBRARY_PATH="$bundle/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

if [ "${RAG_BOOTSTRAP:-}" = 1 ]; then bootstrap; fi

case "$ROLE" in
  monolith) serve_workflow_app monolith monolith ;;
  workflow) serve_workflow_app split-reranker split-reranker split-all ;;
  ovms-reranker) serve_ovms config_reranker.json ;;
  ovms-embedder) serve_ovms config_embedder.json ;;
  fastapi-reranker) serve_fastapi_model reranker ;;
  fastapi-embedder) serve_fastapi_model embedder ;;
  *) die "unknown RAG_ROLE '$ROLE'; expected monolith, workflow, ovms-reranker, ovms-embedder, fastapi-reranker or fastapi-embedder" ;;
esac
