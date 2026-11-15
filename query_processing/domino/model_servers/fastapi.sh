# shellcheck shell=bash
# The fastapi model-server kind: rag-model-server, our FastAPI model server on
# OpenVINO Model Server's /v3 routes and formats. Sourced by domino/app.sh for
# RAG_MODEL_SERVER_KIND=fastapi (registered by
# src/rag_load_test/deploy/model_server_kinds/fastapi_kind.py), it defines the two
# functions every kind's script defines.
#
# It serves the component from that component's local backend, whatever the
# workflow file says; the in-process model exports itself into
# RAG_LOCAL_MODELS__OPENVINO_DIR on its first start (RAG_LOCAL_MODELS__RUNTIME=openvino).

# bootstrap_model_server: this project, as the workflow app installs it.
bootstrap_model_server() {
  install_this_project
}

# serve_model_server <component>: rag-model-server for that component.
serve_model_server() {
  exec rag-model-server --serve "$1" --host 0.0.0.0 --port 8888
}
