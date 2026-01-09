# ================================
# Base image
# ================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ================================
# Environment
# ================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ---- CUDA / PyTorch stability tweaks (CRITICAL) ----
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# ================================
# Model configuration
# ================================
ENV BASE_MODEL=TheBloke/Llama-3.1-70B-Instruct-AWQ
ENV LORA_NAME=pitinf
ENV LORA_REPO=benstaf/pitinf-identity-lora-20260108_162425
ENV PORT=8000

# ================================
# System deps
# ================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Python deps
# ================================
RUN pip3 install --upgrade pip

RUN pip3 install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install \
    vllm \
    transformers \
    accelerate \
    huggingface_hub \
    autoawq

# ================================
# Runtime
# ================================
EXPOSE 8000

CMD ["bash", "-lc", "\
echo '=== GPU INFO ===' && nvidia-smi && \
echo '=== STARTING vLLM OPENAI SERVER (70B AWQ + LoRA) ===' && \
python3 -u -m vllm.entrypoints.openai.api_server \
  --model ${BASE_MODEL} \
  --quantization awq \
  --enable-lora \
  --lora-modules ${LORA_NAME}=${LORA_REPO} \
  --max-lora-rank 64 \
  --host 0.0.0.0 \
  --port ${PORT} \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
"]
