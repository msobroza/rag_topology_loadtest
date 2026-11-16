#!/usr/bin/env bash
# Domino App / Agent entry point for every role of the rag_load_test topologies.
# Pick the role with RAG_ROLE (a Project or App environment variable):
#   monolith          (default) the workflow app for RAG_TOPOLOGY=monolith (the default)
#   workflow          the workflow app for RAG_TOPOLOGY=split-reranker (the default) or
#                     split-all
#   ovms-reranker     OpenVINO Model Server serving the reranker (config_reranker.json)
#   ovms-embedder     OpenVINO Model Server serving the embedder (config_embedder.json)
#                     (both exported in RAG_OVMS_MODELS_DIR, /mnt/data/ovms-models by
#                     default, and served from a folder of the App's own; see below)
#   ovms-generator    OpenVINO Model Server serving the generator, which OVMS pulls
#                     from the Hugging Face Hub itself into RAG_OVMS_MODELS_DIR (no
#                     export; see "The generator" below)
#   fastapi-reranker / fastapi-embedder   the same models behind rag-model-server
#                     (same /v3 routes and formats as OVMS) for the FastAPI-vs-OVMS experiment
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
# them, then reused. An ovms-reranker / ovms-embedder role whose config is missing
# from RAG_OVMS_MODELS_DIR runs ovms/export_models.sh there (it exports both
# models; needs uv and the Hub): the exporter's pinned requirements go into a
# virtualenv of their own ($HOME/ovms-exporter-venv) so they cannot downgrade the
# project's packages, and the project is uv-installed into $HOME/rag-venv first
# when `rag-served-models` is not on PATH (the OVMS roles' bootstrap installs no
# project). The in-process models (monolith, workflow, fastapi-*) export
# themselves into RAG_OPENVINO_MODELS_DIR on their first start
# (RAG_LOCAL_MODEL_RUNTIME=openvino).
#
# RAG_OPENVINO_WEIGHT_FORMAT (fp32, the default, or int8) is what both exports
# store the weights as. An OVMS models folder holds one export, of one weight
# format, recorded in its weight-format file (a folder without one holds int8).
# An exported OVMS role whose folder holds another format stops before OVMS
# starts: set RAG_OPENVINO_WEIGHT_FORMAT to the format found, or point
# RAG_OVMS_MODELS_DIR at another folder, where the asked format is exported on
# first start.
#
# An exported OVMS role then starts ovms on the config `rag-ovms-config` prints: it
# calibrates OVMS's streams and threads left at 0 by RAG_OVMS_NUM_STREAMS and
# RAG_OVMS_THREADS (one ovms start per candidate), and writes, in a fresh
# temporary folder, the export's config and graph with the choice; the export
# itself is never written. When `rag-ovms-config` is not on PATH the project is
# uv-installed into $HOME/rag-venv first; without uv the export's own config is
# served as exported (one stream, no calibration), which a pinned RAG_OVMS_* refuses.
# Pin both variables (rag-calibrate --ovms-models prints them) to skip calibration.
#
# The generator: ovms-generator evaluates `rag-served-models` for
# GENERATOR_SOURCE_MODEL (RAG_OVMS_GENERATION_MODEL), GENERATOR_MODEL_NAME
# (RAG_GENERATION_API_MODEL, the name the workflow asks for),
# GENERATOR_DYNAMIC_SPLIT_FUSE and GENERATOR_MAX_BATCHED_TOKENS
# (RAG_OVMS_GENERATION_DYNAMIC_SPLIT_FUSE / _MAX_BATCHED_TOKENS, OVMS's own
# --dynamic_split_fuse and --max_num_batched_tokens; each left out at OVMS's
# default, true and 0, so only `--dynamic_split_fuse false` is ever passed),
# installing the project into $HOME/rag-venv first with uv when the command is
# missing; with neither it stops. It creates RAG_OVMS_MODELS_DIR (stopping when
# it cannot); OVMS pulls the model there on its first start and reuses it after;
# nothing is calibrated, and the weight-format record is not read. It runs on the
# generic environment with RAG_BOOTSTRAP=1 and RAG_OVMS_BUNDLE, not on the
# prebuilt `ovms` image.
#
# Both workflow app roles run `rag-serve --domino`: Python resolves the embedding and
# rerank backends from RAG_TOPOLOGY (src/rag_load_test/domain/topology.py), requires
# RAG_EMBEDDING_API_URL / RAG_RERANK_API_URL for every component on a model server
# (rag-deploy sets them) and defaults its RAG_*_API_KEY to domino.
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
# The default of RagSettings.openvino_weight_format, repeated in bash; a test
# (tests/scripts/test_domino_app_sh_weight_format.py) fails if the two differ.
WEIGHT_FORMAT=${RAG_OPENVINO_WEIGHT_FORMAT:-fp32}
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

# serve_ovms <part>: serve the reranker or the embedder with OpenVINO Model Server.
serve_ovms() {
  local part=$1 config="$MODELS_DIR/config_$1.json"
  if [ ! -f "$config" ]; then provision_ovms_models "$config"; else require_ovms_weight_format; fi
  command -v ovms >/dev/null 2>&1 \
    || die "'ovms' binary not on PATH; use the environment built from domino/Dockerfile --target ovms"
  # Installed here, not inside the command substitution below: its output must
  # not land in the config path, and its PATH change must last.
  if ! command -v rag-ovms-config >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
    install_project "$HOME/rag-venv" . >&2
  fi
  # set -e stops the App when the substitution fails (die only ends the subshell).
  config=$(ovms_serving_config "$part" "$config")
  run_ovms --config_path "$config"
}

# serve_ovms_generator: serve the generator OVMS pulls from the Hub itself into
# $MODELS_DIR (no export of ours); rag-served-models names it and its options.
serve_ovms_generator() {
  # The prebuilt image built from domino/Dockerfile --target ovms cannot run this Role.
  command -v ovms >/dev/null 2>&1 \
    || die "'ovms' binary not on PATH; run this Role on the generic environment with RAG_BOOTSTRAP=1 and RAG_OVMS_BUNDLE set to the OVMS bundle"
  if ! command -v rag-served-models >/dev/null 2>&1; then
    command -v uv >/dev/null 2>&1 \
      || die "'rag-served-models' is not on PATH and 'uv' is not there to install this project; install the project, or uv"
    install_project "$HOME/rag-venv" . >&2
  fi
  # Assigned first so a failing command stops the App (eval "$(...)" would hide it).
  local served_models
  served_models=$(rag-served-models)
  eval "$served_models"
  # A rag-served-models older than the generator's settings prints none of it.
  [ -n "${GENERATOR_SOURCE_MODEL:-}" ] \
    || die "rag-served-models printed no GENERATOR_SOURCE_MODEL; reinstall this project so the command is current"
  local args=(--source_model "$GENERATOR_SOURCE_MODEL"
    --model_name "$GENERATOR_MODEL_NAME" --model_repository_path "$MODELS_DIR"
    --task text_generation --target_device CPU)
  # Each flag only when it differs from OVMS's own default (split fuse on, 0
  # batched tokens), so the default command line is the one run on a machine.
  if [ "$GENERATOR_DYNAMIC_SPLIT_FUSE" = false ]; then
    args+=(--dynamic_split_fuse false)
  fi
  if [ -n "$GENERATOR_MAX_BATCHED_TOKENS" ]; then
    args+=(--max_num_batched_tokens "$GENERATOR_MAX_BATCHED_TOKENS")
  fi
  # Made here: on the real check the folder existed before OVMS started, so
  # nobody has seen OVMS create it itself.
  mkdir -p "$MODELS_DIR" \
    || die "cannot create the generator's models folder $MODELS_DIR (RAG_OVMS_MODELS_DIR); point RAG_OVMS_MODELS_DIR at a folder this App can write"
  run_ovms "${args[@]}"
}

# run_ovms <ovms argument>...: start ovms with them on 0.0.0.0:8888, or behind
# nginx on the prefix proxy (RAG_OVMS_PREFIX_PROXY=1).
run_ovms() {
  if [ "${RAG_OVMS_PREFIX_PROXY:-}" != 1 ]; then
    exec ovms --rest_port 8888 "$@"
  fi
  command -v nginx >/dev/null 2>&1 \
    || die "RAG_OVMS_PREFIX_PROXY=1 needs nginx (installed by domino/Dockerfile --target ovms)"
  local conf=/tmp/ovms-proxy.conf
  render_proxy_config "$conf"
  ovms --rest_bind_address 127.0.0.1 --rest_port "$OVMS_UPSTREAM_PORT" "$@" &
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

# require_ovms_weight_format: the export in $MODELS_DIR is of the weight format
# asked. ovms/export_models.sh records it in weight-format; a folder without that
# file was exported before it did, always at int8.
require_ovms_weight_format() {
  local found=int8 record="$MODELS_DIR/weight-format"
  if [ -f "$record" ]; then found=$(<"$record"); fi
  [ "$found" = "$WEIGHT_FORMAT" ] \
    || die "$MODELS_DIR (RAG_OVMS_MODELS_DIR) holds an OVMS export at weight format $found, but $WEIGHT_FORMAT was asked (RAG_OPENVINO_WEIGHT_FORMAT, fp32 when unset); set RAG_OPENVINO_WEIGHT_FORMAT=$found to serve that export, or point RAG_OVMS_MODELS_DIR at another folder, where $WEIGHT_FORMAT is exported on first start"
}

# ovms_serving_config <part> <export config>: print the config ovms starts with.
ovms_serving_config() {
  if ! command -v rag-ovms-config >/dev/null 2>&1; then
    # A pin that cannot be applied must not be silently ignored.
    [ -z "${RAG_OVMS_NUM_STREAMS:-}${RAG_OVMS_THREADS:-}" ] \
      || die "RAG_OVMS_NUM_STREAMS / RAG_OVMS_THREADS are set but cannot be applied: 'rag-ovms-config' is not on PATH and 'uv' is not there to install it; install the project, or unset both"
    echo "app.sh[$ROLE]: 'rag-ovms-config' not on PATH and no 'uv' to install it; serving $2 as exported (one stream, not calibrated)" >&2
    echo "$2"
    return
  fi
  local serving_dir
  serving_dir=$(mktemp -d "${TMPDIR:-/tmp}/rag-ovms-serving.XXXXXX")
  rag-ovms-config --serve "$1" --models-dir "$MODELS_DIR" --serving-dir "$serving_dir" \
    || die "rag-ovms-config could not write the OVMS config for $1 (see above); to skip its calibration, pin both RAG_OVMS_NUM_STREAMS and RAG_OVMS_THREADS (rag-calibrate --ovms-models prints them)"
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
  ovms-reranker) serve_ovms reranker ;;
  ovms-embedder) serve_ovms embedder ;;
  ovms-generator) serve_ovms_generator ;;
  fastapi-reranker) serve_fastapi_model reranker ;;
  fastapi-embedder) serve_fastapi_model embedder ;;
  *) die "unknown RAG_ROLE '$ROLE'; expected monolith, workflow, ovms-reranker, ovms-embedder, ovms-generator, fastapi-reranker or fastapi-embedder" ;;
esac
