# Use CUDA devel base image so nvcc and headers are available for kernel compilation
FROM nvidia/cuda:13.3.1-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
    CPATH="/usr/local/cuda/include:$CPATH" \
    LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"

# Install build prerequisites and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Provision Python 3.11/3.12 virtualenv via uv (avoids Python 3.14 ABI incompatibilities)
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install --no-cache "setuptools<70.0.0" wheel torch ninja && \
    uv pip install --no-cache "freetoken[accel]"

EXPOSE 1919

ENTRYPOINT ["ft", "serve"]
CMD ["--host", "0.0.0.0", "--port", "1919"]
