#!/usr/bin/env bash
# Domino App / Agent entry point of every App rag-deploy publishes. RAG_ROLE says
# which (rag-deploy sets it; a Project or App environment variable works too):
#   workflow      (default) the workflow app: rag-serve --domino
#   model-server  a model server of one component of the workflow file, named by
#                 RAG_MODEL_SERVER_COMPONENT, served by the model-server kind
#                 RAG_MODEL_SERVER_KIND: domino/model_servers/<kind>.sh, sourced
#                 here, holds the kind's bootstrap and its command (fastapi.sh:
#                 rag-model-server; ovms.sh: OpenVINO Model Server). A new kind is
#                 one more script there, and this dispatch does not change.
# Domino only routes traffic to processes listening on 0.0.0.0:8888.
#
# RAG_BOOTSTRAP=1 (set by rag-deploy) first installs what the App needs, for a
# generic environment with only uv: the workflow app uv-installs this project into
# a virtualenv under $HOME, with the extras named by RAG_BOOTSTRAP_EXTRAS (e.g.
# "domino") when set; a model server runs its kind's bootstrap. Unset, nothing is
# installed: the prebuilt `rag` / `ovms` images already have it all.
#
# The workflow app reads everything else from its own variables: each component's
# backend and served URL (rag-deploy sets them, and RAG_TOPOLOGY, passed through
# untouched) and, on Domino, calls each served component with the run token. The
# in-process models export themselves into RAG_LOCAL_MODELS__OPENVINO_DIR on their first
# start (RAG_LOCAL_MODELS__RUNTIME=openvino).
set -euo pipefail

cd "$(dirname "$0")/.."

ROLE=${RAG_ROLE:-workflow}
WHO=$ROLE                          # what die names: the role, or the kind and component
MODEL_SERVERS=domino/model_servers # one launch script per model-server kind

die() {
  echo "app.sh[$WHO]: $*" >&2
  exit 1
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

# install_this_project: this project into $HOME/rag-venv, with the extras an
# Agent asks for: its launch script sets RAG_BOOTSTRAP_EXTRAS=domino (tracing).
install_this_project() {
  local project=.
  if [ -n "${RAG_BOOTSTRAP_EXTRAS:-}" ]; then project=".[$RAG_BOOTSTRAP_EXTRAS]"; fi
  install_project "$HOME/rag-venv" "$project"
}

serve_workflow_app() {
  if [ "${RAG_BOOTSTRAP:-}" = 1 ]; then install_this_project; fi
  exec rag-serve --host 0.0.0.0 --port 8888 --domino
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
  local component=${RAG_MODEL_SERVER_COMPONENT:-} kind=${RAG_MODEL_SERVER_KIND:-}
  WHO="$ROLE $kind $component"
  [ -n "$component" ] \
    || die "RAG_MODEL_SERVER_COMPONENT is not set: set it to the component of the workflow file this App serves, such as reranker"
  # A kind names a file here, so nothing but a plain name is looked up.
  case "$kind" in
    "" | *[!a-z0-9_-]*) die "RAG_MODEL_SERVER_KIND='$kind' is no model-server kind; expected one of: $(known_kinds)" ;;
  esac
  [ -f "$MODEL_SERVERS/$kind.sh" ] \
    || die "RAG_MODEL_SERVER_KIND='$kind' has no launch script $MODEL_SERVERS/$kind.sh; expected one of: $(known_kinds)"
  # shellcheck source=/dev/null  # one of domino/model_servers/*.sh
  . "$MODEL_SERVERS/$kind.sh"
  if [ "${RAG_BOOTSTRAP:-}" = 1 ]; then bootstrap_model_server; fi
  serve_model_server "$component"
}

case "$ROLE" in
  workflow) serve_workflow_app ;;
  model-server) serve_model ;;
  *) die "unknown RAG_ROLE '$ROLE'; expected workflow or model-server" ;;
esac
