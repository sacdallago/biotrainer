# This location of python in venv-build needs to match the location in the runtime image,
# so we're manually installing the required python environment
FROM ubuntu:22.04 as venv-build

# build-essential is for jsonnet
RUN apt-get update && \
    apt-get install -y curl build-essential python3 python3-pip python3-distutils python3-venv python3-dev python3-virtualenv git && \
    curl -sSL https://install.python-poetry.org/ | python3 - --version 1.4.2

COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
WORKDIR /app

RUN python3 -m venv .venv && \
    # Install a recent version of pip, otherwise the installation of many linux2010 packages will fail
    .venv/bin/pip install -U pip && \
    # Make sure poetry install the metadata for biotrainer
    mkdir biotrainer && \
    touch biotrainer/__init__.py && \
    touch README.md && \
    $HOME/.local/bin/poetry config virtualenvs.in-project true && \
    $HOME/.local/bin/poetry install --no-dev

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y python3 python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Workaround for when switching the docker user
# https://github.com/numba/numba/issues/4032#issuecomment-547088606
RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -m 777 /.cache && \
    mkdir -m 777 /.cache/bio_embeddings/

COPY --from=venv-build /app/.venv /app/.venv
COPY . /app/

WORKDIR /app

ENTRYPOINT ["/app/.venv/bin/python", "-m", "biotrainer.utilities.cli"]
