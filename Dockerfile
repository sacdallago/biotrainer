FROM python:3.11.11-slim-bookworm AS venv-build

# Installing poetry
ENV PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
RUN pip install poetry==1.8.5

# Copying and installing dependencies
WORKDIR /app
COPY pyproject.toml ./
RUN touch README.md # Poetry needs an (empty) README file
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y python3 python3-distutils \
    && rm -rf /var/lib/apt/lists/* \

# Using virtualenv from venv-build
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=venv-build ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Workaround for when switching the docker user
# https://github.com/numba/numba/issues/4032#issuecomment-547088606
RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -m 777 /.cache

COPY biotrainer ./biotrainer
COPY run-biotrainer.py ./run-biotrainer.py

ENTRYPOINT ["/app/.venv/bin/python3", "run-biotrainer.py"]
