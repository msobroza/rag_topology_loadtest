# shellcheck shell=bash
# The ovms model-server kind: OpenVINO Model Server. Sourced by domino/app.sh for
# QP_MODEL_SERVER_KIND=ovms (registered by
# src/query_processing/deploy/model_server_kinds/ovms_kind.py), it defines the two
# functions every kind's script defines. It serves one component, exported as
# config_<component>.json in QP_OVMS_MODELS_DIR (/mnt/data/ovms-models by default),
# from a folder of the App's own (see below).
#
# Bootstrap (QP_BOOTSTRAP=1): apt-get libxml2 and curl (plus nginx with
# QP_OVMS_PREFIX_PROXY=1, via sudo unless root), and run ovms from QP_OVMS_BUNDLE
# (a folder with bin/ovms and lib/, or its .tar.gz), which qp-deploy sets. It
# installs no project: the prebuilt `ovms` image has ovms on PATH.
#
# Models are provisioned on the App's own disk the first time it starts without
# them, then reused. A config missing from QP_OVMS_MODELS_DIR makes it run
# ovms/export_models.sh there, which exports every embedder and reranker of the
# use case's workflow file (app.sh has entered its directory), a
# config_<component>.json each (needs uv and the Hub): the exporter's pinned
# requirements go into a virtualenv of their own ($HOME/ovms-exporter-venv) so
# they cannot downgrade the project's packages, and the toolkit and the use case
# are uv-installed into $HOME/qp-venv first when `qp-served-models` is not on
# PATH. Every path of the toolkit's own is under $TOOLKIT, which app.sh sets.
#
# The settings' local_models.openvino_weight_format (fp32, the default, or int8) is
# what both exports store the weights as. An OVMS models folder holds one export,
# of one weight format, recorded in its weight-format file (a folder without one
# holds int8). An App whose folder holds another format stops before OVMS starts:
# set QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT to the format found, or point
# QP_OVMS_MODELS_DIR at another folder, where the asked format is exported on
# first start. Who checks depends on who can know what is asked. When the variable
# is set it is what the settings read first, so this script checks at once. Unset,
# the workflow file or .env may state the format, which only Python reads: the
# check is then qp-ovms-config's, with the settings. Without that command nothing
# reads the settings, and this script checks against the default.
#
# The App then starts ovms on the config `qp-ovms-config` prints: it calibrates
# OVMS's streams and threads left at 0 by QP_OVMS_NUM_STREAMS and QP_OVMS_THREADS
# (one ovms start per candidate), and writes, in a fresh temporary folder, the
# export's config and graph with the choice; the export itself is never written.
# When `qp-ovms-config` is not on PATH the toolkit and the use case are
# uv-installed into $HOME/qp-venv first; without uv the export's own config is
# served as exported (one stream, no calibration), which a pinned QP_OVMS_NUM_STREAMS
# or QP_OVMS_THREADS refuses. Pin both variables in this App's block of deploy.yaml
# (qp-calibrate --ovms-models prints them) to skip calibration.
#
# OVMS has no base-path option. When Domino forwards the app path prefix
# (DOMINO_RUN_HOST_PATH) to the process instead of stripping it, set
# QP_OVMS_PREFIX_PROXY=1: OVMS then listens on 127.0.0.1:9000 and nginx on
# 0.0.0.0:8888 strips the prefix (settings: domino/ovms-proxy.conf). Check first with
#   curl -i -H "Authorization: Bearer $(curl -s localhost:8899/access-token)" <App URL>/v3/models/<alias>
# (200 = prefix stripped, no proxy needed; 404 = enable the proxy).

OVMS_MODELS_DIR=${QP_OVMS_MODELS_DIR:-/mnt/data/ovms-models}
OVMS_UPSTREAM_PORT=9000

bootstrap_model_server() {
  local bundle=${QP_OVMS_BUNDLE:-} packages="libxml2 curl" sudo=""
  [ -n "$bundle" ] && [ -e "$bundle" ] \
    || die "QP_OVMS_BUNDLE='$bundle' not found; set it to the OVMS bundle (a folder with bin/ovms and lib/, or its .tar.gz)"
  if [ "${QP_OVMS_PREFIX_PROXY:-}" = 1 ]; then packages="$packages nginx"; fi
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

# serve_model_server <component>: serve that component with OpenVINO Model Server.
serve_model_server() {
  local component=$1 config="$OVMS_MODELS_DIR/config_$1.json"
  if [ ! -f "$config" ]; then
    ovms_provision_models "$config"
  elif [ -n "${QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT:-}" ]; then
    # Set, the variable is what the settings ask: checked before anything else.
    ovms_require_weight_format
  fi
  command -v ovms >/dev/null 2>&1 \
    || die "'ovms' binary not on PATH; use the environment built from domino/Dockerfile --target ovms"
  # Installed here, not inside the command substitution below: its output must
  # not land in the config path, and its PATH change must last.
  if ! command -v qp-ovms-config >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
    install_this_project >&2
  fi
  # set -e stops the App when the substitution fails (die only ends the subshell).
  config=$(ovms_serving_config "$component" "$config")
  if [ "${QP_OVMS_PREFIX_PROXY:-}" != 1 ]; then
    exec ovms --rest_port 8888 --config_path "$config"
  fi
  command -v nginx >/dev/null 2>&1 \
    || die "QP_OVMS_PREFIX_PROXY=1 needs nginx (installed by domino/Dockerfile --target ovms)"
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

# ovms_require_weight_format: the export in $OVMS_MODELS_DIR is of the weight
# format the variable asks, or the default when it is unset.
# ovms/export_models.sh records the format in weight-format; a folder without that
# file was exported before it did, always at int8. The default is that of
# LocalModelsSettings.openvino_weight_format, repeated here for the App with no
# command to ask the settings; a test
# (tests/scripts/test_domino_app_sh_weight_format.py) fails if the two differ.
ovms_require_weight_format() {
  local asked=${QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT:-fp32}
  local found=int8 record="$OVMS_MODELS_DIR/weight-format"
  if [ -f "$record" ]; then found=$(<"$record"); fi
  [ "$found" = "$asked" ] \
    || die "$OVMS_MODELS_DIR (QP_OVMS_MODELS_DIR) holds an OVMS export at weight format $found, but $asked was asked (QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT, fp32 when unset); set QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT=$found to serve that export, or point QP_OVMS_MODELS_DIR at another folder, where $asked is exported on first start"
}

# ovms_serving_config <component> <export config>: print the config ovms starts with.
ovms_serving_config() {
  if ! command -v qp-ovms-config >/dev/null 2>&1; then
    # No command reads the settings: the weight format is checked here, against
    # the variable, else the default.
    ovms_require_weight_format
    # A pin that cannot be applied must not be silently ignored.
    [ -z "${QP_OVMS_NUM_STREAMS:-}${QP_OVMS_THREADS:-}" ] \
      || die "QP_OVMS_NUM_STREAMS / QP_OVMS_THREADS are set but cannot be applied: 'qp-ovms-config' is not on PATH and 'uv' is not there to install it; install the toolkit and the use case, or unset both"
    echo "app.sh[$WHO]: 'qp-ovms-config' not on PATH and no 'uv' to install it; serving $2 as exported (one stream, not calibrated)" >&2
    echo "$2"
    return
  fi
  local serving_dir
  serving_dir=$(mktemp -d "${TMPDIR:-/tmp}/qp-ovms-serving.XXXXXX")
  qp-ovms-config --serve "$1" --models-dir "$OVMS_MODELS_DIR" --serving-dir "$serving_dir" \
    || die "qp-ovms-config could not write the OVMS config for $1: its reason is on the line above; if its calibration is what failed, pinning both QP_OVMS_NUM_STREAMS and QP_OVMS_THREADS (qp-calibrate --ovms-models prints them) skips it"
}

# Render domino/ovms-proxy.conf: __PREFIX__ becomes DOMINO_RUN_HOST_PATH (the
# line is dropped when there is no prefix so "location /" is not duplicated).
ovms_render_proxy_config() {
  local prefix=${DOMINO_RUN_HOST_PATH:-}
  prefix=${prefix%/}
  if [ -n "$prefix" ]; then
    sed "s#__PREFIX__#$prefix#g" "$TOOLKIT/domino/ovms-proxy.conf" > "$1"
  else
    sed '/__PREFIX__/d' "$TOOLKIT/domino/ovms-proxy.conf" > "$1"
  fi
}

# ovms_provision_models <config>: export the models into $OVMS_MODELS_DIR, once.
# Each App has its own disk, so no other process writes this folder: no lock.
ovms_provision_models() {
  local config=$1 exporter_venv="$HOME/ovms-exporter-venv"
  command -v uv >/dev/null 2>&1 \
    || die "OVMS config not found: $config, and 'uv' is not on PATH to export it with ovms/export_models.sh; install uv, or mount a folder holding it and set QP_OVMS_MODELS_DIR"
  echo "app.sh[$WHO]: $config not found; exporting the models into $OVMS_MODELS_DIR (QP_OVMS_MODELS_DIR) with ovms/export_models.sh" >&2
  # export_models.sh asks the project's qp-served-models which models to export.
  if ! command -v qp-served-models >/dev/null 2>&1; then
    install_this_project
  fi
  # Its pinned requirements are pip-installed into $PYTHON, hence a virtualenv of
  # their own; --seed because the script runs "$PYTHON -m pip".
  if [ ! -x "$exporter_venv/bin/python" ]; then
    uv venv "$exporter_venv" --seed --python "$(cat "$PYTHON_VERSION_FILE")"
  fi
  PYTHON="$exporter_venv/bin/python" "$TOOLKIT/ovms/export_models.sh" "$OVMS_MODELS_DIR" \
    || die "exporting the OVMS models into $OVMS_MODELS_DIR failed (ran ovms/export_models.sh with PYTHON=$exporter_venv/bin/python; it needs the Hugging Face Hub and the OVMS exporter's requirements); $config is missing; fix the error above, or mount a folder holding it and set QP_OVMS_MODELS_DIR"
  [ -f "$config" ] \
    || die "ovms/export_models.sh finished but $config is still missing: it exports each embedder and reranker of the workflow file, so check that the component is one, and QP_OVMS_MODELS_DIR=$OVMS_MODELS_DIR"
}
