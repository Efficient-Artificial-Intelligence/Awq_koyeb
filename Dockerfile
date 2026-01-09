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

# REQUIRED: Hugging Face token (set at runtime or here)
# ENV HF_TOKEN=hf_xxxxxxxxxxxxxxxxx

# ================================
# Model configuration
# ================================
ENV BASE_MODEL=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
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
    autoawq \
    torch-c-dlpack-ext

# ================================
# Runtime
# ================================
EXPOSE 8000

CMD ["bash", "-lc", "\
echo '=== GPU INFO ===' && nvidia-smi && \
echo '=== STARTING vLLM OPENAI SERVER (70B AWQ INT4 + LoRA) ===' && \
python3 -u -m vllm.entrypoints.openai.api_server \
  --model ${BASE_MODEL} \
  --quantization awq_marlin \
  --enable-lora \
  --lora-modules ${LORA_NAME}=${LORA_REPO} \
  --max-lora-rank 32 \
  --host 0.0.0.0 \
  --port ${PORT} \
  --gpu-memory-utilization 0.78 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-cudagraph-capture-size 1 \
"]
