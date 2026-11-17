#!/usr/bin/env bash
# Domino App / Agent entry point of every App qp-deploy publishes. QP_ROLE says
# which (qp-deploy sets it; a Project or App environment variable works too):
#   workflow      (default) the workflow app: qp-serve --domino
#   model-server  a model server of one component of the workflow file, named by
#                 QP_MODEL_SERVER_COMPONENT, served by the model-server kind
#                 QP_MODEL_SERVER_KIND: domino/model_servers/<kind>.sh, sourced
#                 here, holds the kind's bootstrap and its command (fastapi.sh:
#                 qp-model-server; ovms.sh: OpenVINO Model Server, on the config
#                 qp-ovms-config prints, its streams and threads calibrated
#                 first). A new kind is one more script there, and this dispatch
#                 does not change.
# Domino only routes traffic to processes listening on 0.0.0.0:8888.
#
# QP_USE_CASE_DIR (set by qp-deploy: the directory it ran from) is the use
# case's directory: the role's command runs there, where it finds the use case's
# workflow file, for the workflow app and for a model server of any kind alike.
# Unset, the command runs in the directory app.sh was started in.
#
# QP_BOOTSTRAP=1 (set by qp-deploy) first installs what the App needs, for a
# generic environment with only uv: the workflow app uv-installs the toolkit, with
# the extras named by QP_BOOTSTRAP_EXTRAS (e.g. "domino") when set, and the use
# case of QP_USE_CASE_DIR, in one command, into a virtualenv under $HOME; a
# model server runs its kind's bootstrap. Unset, nothing is installed: the
# prebuilt `rag` / `ovms` images already have it all.
#
# The workflow app reads everything else from its own variables: each component's
# backend and served URL, and QP_TOPOLOGY_ALIAS, only a label (qp-deploy sets
# them all; passed through untouched), and, on Domino, calls each served
# component with the run token. The in-process models export themselves into
# QP_LOCAL_MODELS__OPENVINO_DIR on their first start
# (QP_LOCAL_MODELS__RUNTIME=openvino).
set -euo pipefail

# The toolkit's directory, query_processing/ in the repository: what is installed
# with the use case, and where every path of the toolkit's own below starts,
# absolute, since the role's command runs in the use case's directory. The
# project's .python-version is one level up, at the repository's root.
TOOLKIT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_VERSION_FILE=$TOOLKIT/../.python-version

ROLE=${QP_ROLE:-workflow}
WHO=$ROLE                                   # what die names: the role, or the kind and component
MODEL_SERVERS=$TOOLKIT/domino/model_servers # one launch script per model-server kind

die() {
  echo "app.sh[$WHO]: $*" >&2
  exit 1
}

# install_project <venv> <requirement>...: uv-install the requirements in one
# command, then put the venv on PATH.
install_project() {
  local venv=$1
  shift
  # Its own virtualenv, on the project's Python (.python-version), which uv
  # fetches when the environment lacks it: the environment's own packages
  # (an old urllib3 in Domino's standard image) break ours.
  uv venv "$venv" --python "$(cat "$PYTHON_VERSION_FILE")"
  uv pip install --python "$venv/bin/python" "$@"
  export PATH="$venv/bin:$PATH"
}

# install_this_project: the toolkit, with the extras an Agent asks for (its
# launch script sets QP_BOOTSTRAP_EXTRAS=domino, for tracing), then the use case
# of QP_USE_CASE_DIR when there is one, into $HOME/qp-venv in one command: the
# use case depends on the toolkit, which no package index has.
install_this_project() {
  local toolkit=$TOOLKIT
  if [ -n "${QP_BOOTSTRAP_EXTRAS:-}" ]; then toolkit="${TOOLKIT}[$QP_BOOTSTRAP_EXTRAS]"; fi
  if [ -n "${QP_USE_CASE_DIR:-}" ]; then
    install_project "$HOME/qp-venv" "$toolkit" "$QP_USE_CASE_DIR"
  else
    install_project "$HOME/qp-venv" "$toolkit"
  fi
}

# enter_use_case: the use case's directory, where the role's command finds the
# use case's files; without QP_USE_CASE_DIR, the directory app.sh started in.
enter_use_case() {
  [ -n "${QP_USE_CASE_DIR:-}" ] || return 0
  cd "$QP_USE_CASE_DIR" 2>/dev/null \
    || die "QP_USE_CASE_DIR='$QP_USE_CASE_DIR' is no directory; expected the use case's directory, as qp-deploy sets it"
}

serve_workflow_app() {
  if [ "${QP_BOOTSTRAP:-}" = 1 ]; then install_this_project; fi
  enter_use_case
  exec qp-serve --host 0.0.0.0 --port 8888 --domino
}

# known_kinds: every model-server kind with a launch script, for the errors.
known_kinds() {
  local script kinds=""
  for script in "$MODEL_SERVERS"/*.sh; do
    script=${script##*/}
    kinds="$kinds ${script%.sh}"
  done
  echo "${kinds# }"
}

# serve_model: source the kind's launch script, which defines
# bootstrap_model_server and serve_model_server <component>, then run them.
serve_model() {
  local component=${QP_MODEL_SERVER_COMPONENT:-} kind=${QP_MODEL_SERVER_KIND:-}
  WHO="$ROLE $kind $component"
  [ -n "$component" ] \
    || die "QP_MODEL_SERVER_COMPONENT is not set: set it to the component of the workflow file this App serves, such as reranker"
  # A kind names a file here, so nothing but a plain name is looked up.
  case "$kind" in
    "" | *[!a-z0-9_-]*) die "QP_MODEL_SERVER_KIND='$kind' is no model-server kind; expected one of: $(known_kinds)" ;;
  esac
  [ -f "$MODEL_SERVERS/$kind.sh" ] \
    || die "QP_MODEL_SERVER_KIND='$kind' has no launch script domino/model_servers/$kind.sh; expected one of: $(known_kinds)"
  # shellcheck source=/dev/null  # one of domino/model_servers/*.sh
  . "$MODEL_SERVERS/$kind.sh"
  if [ "${QP_BOOTSTRAP:-}" = 1 ]; then bootstrap_model_server; fi
  enter_use_case
  serve_model_server "$component"
}

case "$ROLE" in
  workflow) serve_workflow_app ;;
  model-server) serve_model ;;
  *) die "unknown QP_ROLE '$ROLE'; expected workflow or model-server" ;;
esac
