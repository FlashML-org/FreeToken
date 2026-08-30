# Running FreeToken with Docker & Docker Compose

This guide explains how to build, configure, and serve LLMs using FreeToken in a GPU-accelerated Docker container with automatic Hugging Face model downloading.

---

## Prerequisites

- **NVIDIA GPU** with Pascal architecture or newer (Hopper, Ada Lovelace, Ampere, Turing supported).
- **NVIDIA Driver**: Recommended `r580+` (or latest CUDA-compatible driver).
- **Docker Engine** (24.0+) & **Docker Compose** (v2+).
- **NVIDIA Container Toolkit** installed and configured as the default Docker runtime.

Verify GPU passthrough is working before proceeding:
```bash
docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu22.04 nvidia-smi
```

---

## Project Structure

Ensure the following files are in your project directory:

```text
.
├── Dockerfile
├── docker-compose.yml
├── .env                  # Optional: override default models & tokens
└── DOCKER_README.md
```

---

## 1. Build the Docker Image

Build the image locally and tag it as `freetoken:latest`:

```bash
docker build -t freetoken:latest .
```

---

## 2. Configuration (`.env`)

Create a `.env` file in the same directory to configure your model and credentials:

```env
# Required only for gated/private models (e.g., meta-llama/Llama-3.1-8B-Instruct)
HF_TOKEN=hf_your_token_here

# Hugging Face model repository ID
MODEL_NAME=Qwen/Qwen3.6-35B-A3B
```

---

## 3. Running with Docker Compose

### Start the Service
```bash
docker compose up -d
```

### View Logs & Download Progress
FreeToken downloads weights directly to the mounted Hugging Face cache on the host (`~/.cache/huggingface`):
```bash
docker compose logs -f freetoken
```

Wait until you see:
```text
API server is ready to serve on 0.0.0.0:1919
```

### Stop the Service
```bash
docker compose down
```

---

## 4. Alternative: Running via `docker run` (CLI)

If you prefer running a standalone container without Compose:

```bash
docker run -d \
  --name freetoken-server \
  --gpus all \
  --ipc=host \
  -p 1919:1919 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME="/root/.cache/huggingface" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  freetoken:latest \
  --model "Qwen/Qwen3.6-35B-A3B" \
  --host 0.0.0.0 \
  --port 1919
```

---

## 5. Testing the API

FreeToken exposes an OpenAI-compatible HTTP API on port `1919`.

### OpenAI-Compatible Chat Completion (`curl`)

```bash
curl [http://127.0.0.1:1919/v1/chat/completions](http://127.0.0.1:1919/v1/chat/completions) \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a quick Python script to calculate Fibonacci numbers."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Python Client Example (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1919/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=[
        {"role": "user", "content": "Explain KV caching in two sentences."}
    ]
)

print(response.choices[0].message.content)
```

---

## 6. Performance & Troubleshooting Notes

- **`ipc: host`**: Required. High-performance LLM engines use shared memory (`/dev/shm`) for multi-worker communication and KV cache transfers. Omitting this can cause immediate `Bus error` or `CUDA OOM` crashes.
- **Persistent Cache**: Model weights are stored in `~/.cache/huggingface` on the host machine. Subsequent runs will load from disk instantly without re-downloading.
- **Custom Model Paths**: To load local safetensors or GGUF files instead of downloading from Hugging Face, mount your local directory into the container (e.g., `-v /path/to/models:/models`) and set `--model /models/your-model-folder`.
