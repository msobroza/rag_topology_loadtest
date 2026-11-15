# Domino 6.2 compute environments for rag_load_test: two build targets, one file.
#
# Build from the repository's root, the context: it holds .python-version, the
# toolkit's directory query_processing/ (pyproject.toml, src/, ovms/) and the use
# case's (USE_CASE, the RAG's by default: its pyproject.toml and workflow.yaml):
#   docker build -f query_processing/domino/Dockerfile --target rag  -t <registry>/rag-load-test-rag  .
#   docker build -f query_processing/domino/Dockerfile --target ovms -t <registry>/rag-load-test-ovms .
#
#   rag   for the workflow apps (RAG_ROLE=workflow), the model servers of kind
#         fastapi (RAG_ROLE=model-server, RAG_MODEL_SERVER_KIND=fastapi) and the
#         Workspaces that run Locust: the toolkit with the [loadtest,domino]
#         extras and the use case, the use case's two HF models pre-downloaded,
#         HF_HUB_OFFLINE=1.
#   ovms  for the model servers of kind ovms (RAG_ROLE=model-server,
#         RAG_MODEL_SERVER_KIND=ovms): the ovms binary tarball pinned in
#         ovms/versions.env, unpacked under /opt/ovms, plus nginx for the
#         optional prefix proxy (RAG_OVMS_PREFIX_PROXY=1).
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

# Only what pip needs, laid out as in the repository (the use case finds the
# toolkit beside it): the entry point (domino/app.sh), locustfile.py and the use
# case's other files come from the project files Domino mounts into the run, and
# .env never enters the image.
COPY .python-version /opt/rag_load_test/
COPY query_processing/pyproject.toml /opt/rag_load_test/query_processing/
COPY query_processing/src /opt/rag_load_test/query_processing/src
COPY ${USE_CASE}/pyproject.toml ${USE_CASE}/workflow.yaml /opt/rag_load_test/${USE_CASE}/
# The toolkit and the use case in one command, the toolkit first: no package index
# has the toolkit. [loadtest] adds locust; [domino] adds dominodatalab[agents] for
# RAG_DOMINO__TRACING=true.
RUN pip install --no-cache-dir uv \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv /opt/rag-venv \
        --python "$(cat /opt/rag_load_test/.python-version)" \
    && uv pip install --no-cache --python /opt/rag-venv/bin/python \
        "/opt/rag_load_test/query_processing[loadtest,domino]" \
        "/opt/rag_load_test/${USE_CASE}"
ENV PATH=/opt/rag-venv/bin:$PATH

# Download the use case's two HF models and convert them to OpenVINO, so app
# start-up needs no network and no conversion. rag-export-openvino takes the
# source ids from the use case's workflow file (the one served-model pin), read
# in its directory as every command reads it, so this step cannot drift from what
# the apps load. The download stays in HF_HOME, so RAG_LOCAL_MODELS__RUNTIME=torch
# also starts offline. Both folders are world-readable because the build runs as
# root but Domino runs apps as ubuntu.
ENV HF_HOME=/opt/hf-cache
ENV RAG_LOCAL_MODELS__OPENVINO_DIR=/opt/rag-models/openvino
RUN cd "/opt/rag_load_test/${USE_CASE}" \
    && rag-export-openvino \
    && chmod -R a+rX /opt/hf-cache /opt/rag-models
ENV HF_HUB_OFFLINE=1

USER ubuntu

# --- ovms --------------------------------------------------------------------
FROM ${BASE_IMAGE} AS ovms
USER root

# curl + libxml2 are the documented runtime prerequisites of the OVMS bare-metal
# tarball; nginx is the optional prefix-stripping proxy (RAG_OVMS_PREFIX_PROXY=1),
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
