# shellcheck shell=bash
# The vllm model-server kind: vLLM's OpenAI-compatible server. Sourced by
# domino/app.sh for QP_MODEL_SERVER_KIND=vllm (registered by
# src/query_processing/deploy/model_server_kinds/vllm_kind.py), it defines the two
# functions every kind's script defines. It serves one component, an embedder, a
# reranker or an LLM, from the weights its source id names, which vLLM fetches
# itself: no export, no calibration, no nginx.
#
# Bootstrap (QP_BOOTSTRAP=1): nothing when the environment already has the vllm
# command. Otherwise uv installs the toolkit's `vllm` dependency group alone (the
# CPU wheel pinned in pyproject.toml, torch's CPU build) into $HOME/vllm-venv: a
# virtualenv of its own, as vLLM pins transformers and torch apart from the
# project's. That wheel needs glibc 2.39 (Ubuntu 24.04): neither image of
# domino/Dockerfile can install it. Intel OpenMP is preloaded when the
# virtualenv holds it, as vLLM's CPU page asks for an install from wheels. When
# serving will install the toolkit (below), the wheel of doclang's core is looked
# for before anything else is installed.
#
# The App then exports the CPU variables `qp-vllm-config --environment` prints
# (QP_VLLM_CPU_OMP_THREADS_BIND, QP_VLLM_CPU_RESERVED_CPUS) and runs
# `vllm serve` on the words `qp-vllm-config --serve` prints: the component's
# source id, --served-model-name and its alias, then the flags of QP_VLLM_*
# (the pass-through QP_VLLM_ARGS last), on 0.0.0.0:8888. Both commands check the
# settings first and stop the App in one line. When `qp-vllm-config` is not on
# PATH the toolkit and the use case are uv-installed into $HOME/qp-venv first;
# with neither it stops. vLLM has a base-path option: Domino's path prefix
# (DOMINO_RUN_HOST_PATH) becomes --root-path, without its trailing slash.

VLLM_SOURCES="use an environment with glibc 2.39 or later (Ubuntu 24.04) and uv, where QP_BOOTSTRAP=1 installs it, or one that already has the vllm command"

# bootstrap_model_server: the toolkit's vllm dependency group alone, into
# $HOME/vllm-venv, unless the environment has the vllm command.
bootstrap_model_server() {
  # Without qp-vllm-config the toolkit is installed at serve time: its wheel is
  # looked for first, before vLLM is installed.
  if ! command -v qp-vllm-config >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
    require_doclang_wheel
  fi
  if command -v vllm >/dev/null 2>&1; then return; fi
  command -v uv >/dev/null 2>&1 \
    || die "'vllm' is not on PATH and 'uv' is not there to install it; $VLLM_SOURCES"
  local venv="$HOME/vllm-venv"
  uv venv "$venv" --python "$(cat "$PYTHON_VERSION_FILE")"
  # In the toolkit's directory, where --group reads pyproject.toml.
  (cd "$TOOLKIT" && uv pip install --python "$venv/bin/python" --group vllm --torch-backend cpu) \
    || die "installing vLLM failed; its CPU wheel needs glibc 2.39 (Ubuntu 24.04) and this environment has $(vllm_glibc_found); use such an environment, or one that already has the vllm command"
  export PATH="$venv/bin:$PATH"
  if [ -f "$venv/lib/libiomp5.so" ]; then
    export LD_PRELOAD="$venv/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
}

# vllm_glibc_found: the first line `ldd --version` prints, or "unknown".
vllm_glibc_found() {
  local found=""
  if command -v ldd >/dev/null 2>&1; then
    found=$(ldd --version 2>/dev/null | head -n 1) || true
  fi
  echo "${found:-unknown}"
}

# serve_model_server <component>: vllm serve on the words qp-vllm-config prints.
serve_model_server() {
  local component=$1
  command -v vllm >/dev/null 2>&1 || die "'vllm' is not on PATH; $VLLM_SOURCES"
  if ! command -v qp-vllm-config >/dev/null 2>&1; then
    command -v uv >/dev/null 2>&1 \
      || die "neither 'qp-vllm-config' (to read the $component's model and the QP_VLLM_* settings) nor 'uv' (to install it) is on PATH; install uv, or the toolkit and the use case"
    install_this_project
  fi
  # Each assigned on its own so that set -e stops the App when the command
  # refuses: its one line on standard error says why.
  local environment_line serve_line
  environment_line=$(qp-vllm-config --environment "$component")
  serve_line=$(qp-vllm-config --serve "$component")
  local -a exported=() words=()
  eval "exported=($environment_line)"  # NAME=value words shell-quoted by qp-vllm-config
  eval "words=($serve_line)"           # words shell-quoted by qp-vllm-config
  local assignment
  for assignment in ${exported[@]+"${exported[@]}"}; do
    export "${assignment?}"  # a NAME=value word: ? quiets shellcheck SC2163
  done
  words+=(--host 0.0.0.0 --port 8888)
  local prefix=${DOMINO_RUN_HOST_PATH:-}
  prefix=${prefix%/}
  if [ -n "$prefix" ]; then words+=(--root-path "$prefix"); fi
  exec vllm serve "${words[@]}"
}
