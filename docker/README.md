# CUDA 13 image

Build the locked FreeToken runtime from the repository root:

```bash
docker build -f docker/Dockerfile.cuda13 -t freetoken:cuda13 .
```

The image uses CUDA 13.0.2, the exact dependency versions in `uv.lock`, and
FlashInfer's CUDA 13 wheels. Run it with an explicit model and a persistent
Hugging Face cache:

```bash
docker run --gpus all -p 1919:1919 \
  -v "$HF_HOME:/models" -e HF_HOME=/models \
  freetoken:cuda13 serve --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --host 0.0.0.0 --port 1919
```

CUDA 13 normally requires driver 580 or newer. On hardware supported by
NVIDIA's forward-compatibility package, a driver 570 deployment can opt in:

```bash
docker build -f docker/Dockerfile.cuda13 \
  --build-arg INSTALL_CUDA_COMPAT=1 \
  -t freetoken:cuda13-compat .
```
