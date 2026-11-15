# shellcheck shell=bash
# The ovms model-server kind: OpenVINO Model Server. Sourced by domino/app.sh for
# RAG_MODEL_SERVER_KIND=ovms (registered by
# src/rag_load_test/deploy/model_server_kinds/ovms_kind.py), it defines the two
# functions every kind's script defines. It serves one component from the config
# config_<component>.json in RAG_OVMS_MODELS_DIR (/mnt/data/ovms-models by default).
#
# Bootstrap (RAG_BOOTSTRAP=1): apt-get libxml2 and curl (plus nginx with
# RAG_OVMS_PREFIX_PROXY=1, via sudo unless root), and run ovms from RAG_OVMS_BUNDLE
# (a folder with bin/ovms and lib/, or its .tar.gz), which rag-deploy sets. It
# installs no project: the prebuilt `ovms` image has ovms on PATH.
#
# Models are provisioned on the App's own disk the first time it starts without
# them, then reused. A config missing from RAG_OVMS_MODELS_DIR makes it run
# ovms/export_models.sh there, which exports every embedder and reranker of the
# workflow file, a config_<component>.json each (needs uv and the Hub): the
# exporter's pinned requirements go into a virtualenv of their own
# ($HOME/ovms-exporter-venv) so they cannot downgrade the project's packages, and
# the project is uv-installed into $HOME/rag-venv first when `rag-served-models` is
# not on PATH.
#
# OVMS has no base-path option. When Domino forwards the app path prefix
# (DOMINO_RUN_HOST_PATH) to the process instead of stripping it, set
# RAG_OVMS_PREFIX_PROXY=1: OVMS then listens on 127.0.0.1:9000 and nginx on
# 0.0.0.0:8888 strips the prefix (settings: domino/ovms-proxy.conf). Check first with
#   curl -i -H "Authorization: Bearer $(curl -s localhost:8899/access-token)" <App URL>/v3/models/<alias>
# (200 = prefix stripped, no proxy needed; 404 = enable the proxy).

OVMS_MODELS_DIR=${RAG_OVMS_MODELS_DIR:-/mnt/data/ovms-models}
OVMS_UPSTREAM_PORT=9000

bootstrap_model_server() {
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

# serve_model_server <component>: ovms on that component's config.
serve_model_server() {
  local config="$OVMS_MODELS_DIR/config_$1.json"
  if [ ! -f "$config" ]; then ovms_provision_models "$config"; fi
  command -v ovms >/dev/null 2>&1 \
    || die "'ovms' binary not on PATH; use the environment built from domino/Dockerfile --target ovms"
  if [ "${RAG_OVMS_PREFIX_PROXY:-}" != 1 ]; then
    exec ovms --rest_port 8888 --config_path "$config"
  fi
  command -v nginx >/dev/null 2>&1 \
    || die "RAG_OVMS_PREFIX_PROXY=1 needs nginx (installed by domino/Dockerfile --target ovms)"
  local conf=/tmp/ovms-proxy.conf
  ovms_render_proxy_config "$conf"
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

# Render domino/ovms-proxy.conf: __PREFIX__ becomes DOMINO_RUN_HOST_PATH (the
# line is dropped when there is no prefix so "location /" is not duplicated).
ovms_render_proxy_config() {
  local prefix=${DOMINO_RUN_HOST_PATH:-}
  prefix=${prefix%/}
  if [ -n "$prefix" ]; then
    sed "s#__PREFIX__#$prefix#g" domino/ovms-proxy.conf > "$1"
  else
    sed '/__PREFIX__/d' domino/ovms-proxy.conf > "$1"
  fi
}

# ovms_provision_models <config>: export the models into $OVMS_MODELS_DIR, once.
# Each App has its own disk, so no other process writes this folder: no lock.
ovms_provision_models() {
  local config=$1 exporter_venv="$HOME/ovms-exporter-venv"
  command -v uv >/dev/null 2>&1 \
    || die "OVMS config not found: $config, and 'uv' is not on PATH to export it with ovms/export_models.sh; install uv, or mount a folder holding it and set RAG_OVMS_MODELS_DIR"
  echo "app.sh[$WHO]: $config not found; exporting the models into $OVMS_MODELS_DIR (RAG_OVMS_MODELS_DIR) with ovms/export_models.sh" >&2
  # export_models.sh asks the project's rag-served-models which models to export.
  if ! command -v rag-served-models >/dev/null 2>&1; then
    install_project "$HOME/rag-venv" .
  fi
  # Its pinned requirements are pip-installed into $PYTHON, hence a virtualenv of
  # their own; --seed because the script runs "$PYTHON -m pip".
  if [ ! -x "$exporter_venv/bin/python" ]; then
    uv venv "$exporter_venv" --seed --python "$(cat ../.python-version)"
  fi
  PYTHON="$exporter_venv/bin/python" ovms/export_models.sh "$OVMS_MODELS_DIR" \
    || die "exporting the OVMS models into $OVMS_MODELS_DIR failed (ran ovms/export_models.sh with PYTHON=$exporter_venv/bin/python; it needs the Hugging Face Hub and the OVMS exporter's requirements); $config is missing; fix the error above, or mount a folder holding it and set RAG_OVMS_MODELS_DIR"
  [ -f "$config" ] \
    || die "ovms/export_models.sh finished but $config is still missing: it exports each embedder and reranker of the workflow file, so check that the component is one, and RAG_OVMS_MODELS_DIR=$OVMS_MODELS_DIR"
}
