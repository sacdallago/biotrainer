FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=100

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv

# Copy only requirements first to leverage Docker caching
COPY pyproject.toml ./
RUN touch README.md

# Install dependencies
RUN uv pip install --system -e .

# Workaround for when switching the docker user
RUN mkdir /tmp/numba_cache && chmod 777 /tmp/numba_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -m 777 /.cache

# Copy application code
COPY biotrainer ./biotrainer
COPY run-biotrainer.py ./run-biotrainer.py

# Remove cache to reduce container size
RUN rm -rf ~/.cache/uv

ENTRYPOINT ["python3", "run-biotrainer.py"]