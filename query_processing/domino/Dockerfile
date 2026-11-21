# Domino 6.2 compute environments for query_processing: two build targets, one file.
#
# Build from the repository's root, the context: it holds .python-version, the
# core's directory query_processing_core/ (pyproject.toml, src/), the toolkit's
# directory query_processing/ (pyproject.toml, src/, ovms/), the use
# case's (USE_CASE, the RAG's by default: its pyproject.toml, its package under
# src/, and the workflow, prompts and routes files its settings read) and
# wheels/, which git ignores: the wheel of doclang's core (doclang-domain-core),
# which the core depends on and no package index has, that `make doclang-wheel`
# at the root builds from its clone. The rag stage copies wheels/ and installs
# doclang's core from it; without it the build stops at that COPY. Build the
# wheel first:
#   make doclang-wheel
#   docker build -f query_processing/domino/Dockerfile --target rag  -t <registry>/query-processing-rag  .
#   docker build -f query_processing/domino/Dockerfile --target ovms -t <registry>/query-processing-ovms .
#
#   rag   for the workflow apps (QP_ROLE=workflow), the model servers of kind
#         fastapi (QP_ROLE=model-server, QP_MODEL_SERVER_KIND=fastapi) and the
#         Workspaces that run Locust: the toolkit with the [loadtest,domino]
#         extras and the use case, each embedder and reranker of its workflow
#         file pre-downloaded,
#         HF_HUB_OFFLINE=1.
#   ovms  for the model servers of kind ovms (QP_ROLE=model-server,
#         QP_MODEL_SERVER_KIND=ovms): the ovms binary tarball pinned in
#         ovms/versions.env, unpacked under /opt/ovms, plus nginx for the
#         optional prefix proxy (QP_OVMS_PREFIX_PROXY=1). Not for an LLM's
#         App: this image holds neither uv nor the toolkit, which that App
#         needs to read the LLM's model and options. It runs on the generic
#         environment qp-deploy publishes into, with QP_BOOTSTRAP=1 and the
#         OVMS bundle (QP_OVMS_BUNDLE).
#
# Neither image is for the model servers of kind vllm (QP_MODEL_SERVER_KIND=vllm):
# both are built on Ubuntu 22.04, where vLLM's CPU wheel does not install (it
# needs glibc 2.39), and neither holds the vllm command. Those Apps run in an
# environment on Ubuntu 24.04 with uv and QP_BOOTSTRAP=1, which installs the
# toolkit's `vllm` dependency group, or in one that already has vllm.
#
# Push the images and create two Domino Environments that use them as custom
# base images, or pick BASE_IMAGE as the base environment and paste one stage's
# instructions into the environment's "Dockerfile Instructions" field.
# Domino's 6.2 standard environment (Ubuntu 22.04). Its own Python is 3.10 and its
# packages clash with ours, so the project gets a virtualenv on the Python of
# .python-version. CI runs `make check` inside this same image
# (.github/workflows/checks.yml reads this line).
ARG BASE_IMAGE=quay.io/domino/domino-standard-environment:ubuntu22-py3.10-r4.5-domino6.2-standard

# The use case the image is built for: a directory beside query_processing/.
ARG USE_CASE=query_processing_rag_usecase

# --- rag ---------------------------------------------------------------------
FROM ${BASE_IMAGE} AS rag
ARG USE_CASE
USER root

# What pip needs and every file the use case's settings read, laid out as in
# the repository (the use case finds the toolkit beside it): its package under
# src/, which its workflow file names under extensions, its workflow file, its
# prompts and its routes. The entry point (domino/app.sh), locustfile.py and the
# use case's other files come from the project files Domino mounts into the
# run, and .env never enters the image. A use case's tests can read these COPY
# lines and read its settings from what they copy, so that a file it gains is
# not forgotten here.
COPY .python-version /opt/qp/
COPY query_processing_core/pyproject.toml /opt/qp/query_processing_core/
COPY query_processing_core/src /opt/qp/query_processing_core/src
COPY query_processing/pyproject.toml /opt/qp/query_processing/
COPY query_processing/src /opt/qp/query_processing/src
COPY ${USE_CASE}/pyproject.toml ${USE_CASE}/workflow.yaml ${USE_CASE}/prompts.yaml ${USE_CASE}/routes.yaml /opt/qp/${USE_CASE}/
COPY ${USE_CASE}/src /opt/qp/${USE_CASE}/src
# The wheel of doclang's core, which `make doclang-wheel` writes into wheels/.
COPY wheels /opt/qp/wheels/
# The core, the toolkit and the use case in one command, in that order, doclang's
# core found among the wheels: no package index has any of them. [loadtest] adds locust and
# rouge-score, for the Workspaces that run the load test and compare answers (an
# App runs neither);
# [domino] adds dominodatalab[agents] for QP_DOMINO__TRACING=true.
RUN pip install --no-cache-dir uv \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv /opt/qp-venv \
        --python "$(cat /opt/qp/.python-version)" \
    && uv pip install --no-cache --python /opt/qp-venv/bin/python \
        --find-links /opt/qp/wheels \
        /opt/qp/query_processing_core \
        "/opt/qp/query_processing[loadtest,domino]" \
        "/opt/qp/${USE_CASE}"
ENV PATH=/opt/qp-venv/bin:$PATH

# Download each embedder and reranker of the use case's workflow file and convert
# them to OpenVINO, so app start-up needs no network and no conversion. qp-export-openvino takes the
# source ids from the use case's workflow file (the one served-model pin), read
# in its directory as every command reads it, so this step cannot drift from what
# the apps load. The download stays in HF_HOME, so QP_LOCAL_MODELS__RUNTIME=torch
# also starts offline. Both folders are world-readable because the build runs as
# root but Domino runs apps as ubuntu.
ENV HF_HOME=/opt/hf-cache
ENV QP_LOCAL_MODELS__OPENVINO_DIR=/opt/qp-models/openvino
RUN cd "/opt/qp/${USE_CASE}" \
    && qp-export-openvino \
    && chmod -R a+rX /opt/hf-cache /opt/qp-models
ENV HF_HUB_OFFLINE=1

USER ubuntu

# --- ovms --------------------------------------------------------------------
FROM ${BASE_IMAGE} AS ovms
USER root

# curl + libxml2 are the documented runtime prerequisites of the OVMS bare-metal
# tarball; nginx is the optional prefix-stripping proxy (QP_OVMS_PREFIX_PROXY=1),
# run as the ubuntu user with every path under /tmp, hence the chown.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libxml2 nginx \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R ubuntu:ubuntu /var/log/nginx /var/lib/nginx

# OVMS_BINARY_URL is read from ovms/versions.env at build time (single pin, no
# drift). The tarball unpacks to ovms/{bin,lib}; /opt keeps PATH below stable.
COPY query_processing/ovms/versions.env /tmp/versions.env
RUN . /tmp/versions.env \
    && curl -fL "$OVMS_BINARY_URL" -o /tmp/ovms.tgz \
    && tar -xzf /tmp/ovms.tgz -C /opt \
    && chmod -R a+rX /opt/ovms \
    && rm -f /tmp/ovms.tgz /tmp/versions.env

ENV PATH=/opt/ovms/bin:$PATH LD_LIBRARY_PATH=/opt/ovms/lib

USER ubuntu
