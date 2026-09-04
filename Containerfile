ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/freetoken

COPY pyproject.toml README.md LICENSE setup.py ./
COPY python ./python

# The development image keeps nvcc available when FreeToken compiles JIT kernels.
RUN python3 -m pip install --no-cache-dir ".[accel]"

EXPOSE 1919

ENTRYPOINT ["ft"]
CMD ["serve"]
