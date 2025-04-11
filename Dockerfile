# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-minimal:9.5-1742914212

FROM ${BASE_IMAGE} AS base

# Enable the Google Cloud CLI repo
COPY axlearn/cloud/gcp/repo/google-cloud-sdk.repo /etc/yum.repos.d/google-cloud-sdk.repo

# Install curl and gpupg first so that we can use them to install google-cloud-cli.
RUN microdnf clean all && \
  microdnf install -y gnupg python3.11 google-cloud-cli wget nano findutils && \
  microdnf clean all && rm -rf /var/cache/yum && \
  cp /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt

# Setup.
RUN mkdir -p /root
WORKDIR /root
# Introduce the minimum set of files for install.
COPY README.md README.md
COPY pyproject.toml pyproject.toml
RUN mkdir axlearn && touch axlearn/__init__.py
# Setup venv to suppress pip warnings.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies and purge the cache
RUN pip install --upgrade pip && pip install flit pytest pytest-instafail && pip cache purge

################################################################################
# CI container spec.                                                           #
################################################################################

# Leverage multi-stage build for unit tests.
FROM base AS ci

# TODO(markblee): Remove gcp,vertexai_tensorboard from CI.
RUN pip install .[core,dev,grain,gcp,vertexai_tensorboard]
COPY . .

# Defaults to an empty string, i.e. run pytest against all files.
ARG PYTEST_FILES=''
# Defaults to empty string, i.e. do NOT skip precommit
ARG SKIP_PRECOMMIT=''
# `exit 1` fails the build.
RUN ./run_tests.sh $SKIP_PRECOMMIT "${PYTEST_FILES}"

################################################################################
# Bastion container spec.                                                      #
################################################################################

FROM base AS bastion

# TODO(markblee): Consider copying large directories separately, to cache more aggressively.
# TODO(markblee): Is there a way to skip the "production" deps?
COPY . /root/
RUN pip install .[core,gcp,vertexai_tensorboard]

################################################################################
# Dataflow container spec.                                                     #
################################################################################

FROM base AS dataflow

# Beam workers default to creating a new virtual environment on startup. Instead, we want them to
# pickup the venv setup above. An alternative is to install into the global environment.
ENV RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1
RUN pip install .[core,gcp,dataflow]
COPY . .

# Dataflow workers can't start properly if the entrypoint is not set
# See: https://cloud.google.com/dataflow/docs/guides/build-container-image#use_a_custom_base_image
COPY --from=apache/beam_python3.11_sdk:2.52.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT ["/opt/apache/beam/boot"]

################################################################################
# TPU container spec.                                                          #
################################################################################

FROM base AS tpu

ARG EXTRAS=

ENV PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Ensure we install the TPU version, even if building locally.
# Jax will fallback to CPU when run on a machine without TPU.
RUN pip install .[core,tpu]
RUN if [ -n "$EXTRAS" ]; then pip install .[$EXTRAS]; fi
COPY . .

################################################################################
# GPU container spec.                                                          #
################################################################################

FROM base AS gpu

# Enable the CUDA repo
RUN curl -o /etc/yum.repos.d/cuda-rhel9.repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install the CUDA development libraries (libnvrtc.so), needed by newer versions of JAX
RUN microdnf install -y cuda-libraries-devel-12-8 libibverbs && \
  microdnf clean all && rm -rf /var/cache/yum

# TODO(markblee): Support extras.
ENV PIP_FIND_LINKS="https://storage.googleapis.com/jax-releases/jax_nightly_releases.html" JAX_TRACEBACK_FILTERING=off
RUN pip install .[core,gpu] && pip cache purge

COPY . .

################################################################################
# Final target spec.                                                           #
################################################################################

FROM ${TARGET} AS final
