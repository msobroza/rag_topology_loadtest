# Domino 6.2 compute environments for rag_load_test: two build targets, one file.
#
# Build from the project root (the directory holding pyproject.toml is the context):
#   docker build -f domino/Dockerfile --target rag  -t <registry>/rag-load-test-rag  .
#   docker build -f domino/Dockerfile --target ovms -t <registry>/rag-load-test-ovms .
#
#   rag   for the RAG apps (RAG_ROLE=monolith | workflow) and for Workspaces that
#         run Locust: this package with the [loadtest,domino] extras, both HF
#         models pre-downloaded, HF_HUB_OFFLINE=1.
#   ovms  for the OVMS apps (RAG_ROLE=ovms-reranker | ovms-embedder): the ovms
#         binary tarball pinned in ovms/versions.env, unpacked under /opt/ovms,
#         plus nginx for the optional prefix proxy (RAG_OVMS_PREFIX_PROXY=1).
#
# Push the images and create two Domino Environments that use them as custom
# base images, or pick BASE_IMAGE as the base environment and paste one stage's
# instructions into the environment's "Dockerfile Instructions" field.
ARG BASE_IMAGE=quay.io/domino/compute-environment-images:ubuntu22-py3.10-r4.4-domino6.2-standard

# --- rag ---------------------------------------------------------------------
FROM ${BASE_IMAGE} AS rag
USER root

# Only what pip needs: the entry point (domino/app.sh) and locustfile.py come
# from the project files Domino mounts into the run, and .env never enters the image.
COPY pyproject.toml README.md /opt/rag_load_test/
COPY src /opt/rag_load_test/src
# [loadtest] adds locust; [domino] adds dominodatalab[agents] for RAG_DOMINO_TRACING=true.
RUN pip install --no-cache-dir "/opt/rag_load_test[loadtest,domino]"

# Pre-download the two HF models so app start-up needs no network. The source ids
# come from the installed package's settings defaults (the one served-model pin),
# so this step cannot drift from what the apps load. HF_HOME is world-readable
# because the build runs as root but Domino runs apps as ubuntu.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; from rag_load_test.settings import RagSettings; s = RagSettings(_env_file=None); SentenceTransformer(s.embedder_model); CrossEncoder(s.reranker_model)" \
    && chmod -R a+rX /opt/hf-cache
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
COPY ovms/versions.env /tmp/versions.env
RUN . /tmp/versions.env \
    && curl -fL "$OVMS_BINARY_URL" -o /tmp/ovms.tgz \
    && tar -xzf /tmp/ovms.tgz -C /opt \
    && chmod -R a+rX /opt/ovms \
    && rm -f /tmp/ovms.tgz /tmp/versions.env

ENV PATH=/opt/ovms/bin:$PATH LD_LIBRARY_PATH=/opt/ovms/lib

USER ubuntu
