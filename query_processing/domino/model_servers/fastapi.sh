# shellcheck shell=bash
# The fastapi model-server kind: qp-model-server, our FastAPI model server on
# OpenVINO Model Server's /v3 routes and formats. Sourced by domino/app.sh for
# QP_MODEL_SERVER_KIND=fastapi (registered by
# src/query_processing/deploy/model_server_kinds/fastapi_kind.py), it defines the two
# functions every kind's script defines.
#
# It serves the component from that component's local backend, whatever the
# workflow file says; the in-process model exports itself into
# QP_LOCAL_MODELS__OPENVINO_DIR on its first start (QP_LOCAL_MODELS__RUNTIME=openvino).

# bootstrap_model_server: the toolkit and the use case, as the workflow app
# installs them.
bootstrap_model_server() {
  install_this_project
}

# serve_model_server <component>: qp-model-server for that component.
serve_model_server() {
  exec qp-model-server --serve "$1" --host 0.0.0.0 --port 8888
}
