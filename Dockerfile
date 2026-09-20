FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04

RUN userdel -r ubuntu \
    && useradd -m -u 1000 app \
    && mkdir -p /app && chown -R app:app /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/home/app/.local/bin:$PATH"

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-wheel \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER app

WORKDIR /app

RUN git clone https://github.com/FlashML-org/FreeToken.git .

RUN python3.12 -m venv .venv

RUN .venv/bin/python -m pip install -e ".[accel]"

CMD [".venv/bin/python", "-m", "ft", "serve", "--model", "nvidia/Qwen3.6-35B-A3B-NVFP4"]