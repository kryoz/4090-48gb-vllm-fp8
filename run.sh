#!/bin/bash

CTX_SIZE=auto
# 2026-10-04: KV cache в bf16 (lossless) вместо fp8_e4m3: пул VRAM ~215K токенов,
# ёмкость компенсируется оффлоадом холодных префиксов в RAM (native backend).
QUANT="bfloat16"
QUANT_CONFIG=(--dtype bfloat16 --kv-cache-dtype $QUANT --max-model-len $CTX_SIZE)
OFFLOAD_GIB=24
OFFLOAD_CONFIG=(--kv-offloading-size $OFFLOAD_GIB --kv-offloading-backend native)

PERF_MODE="balanced"
MEM=0.95
NUM_SEQS=6

SPEC_MTP='{"method": "mtp", "num_speculative_tokens": 3}' 
SPEC_CONFIG=(-sc "$SPEC_MTP") 
MISC_CONFIG=(--async-scheduling --enable-prefix-caching --enable-auto-tool-choice --enable-chunked-prefill --mamba-cache-mode align --block-size 32 --enable-flashinfer-autotune)
# --attention-backend flashinfer --enable-flashinfer-autotune \
 
docker run --rm --name vllm --runtime nvidia --gpus all \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  --cpuset-cpus 0-3 \
  -v /root/.cache/vllm:/root/.cache/vllm \
  -v /root/.triton:/root/.triton \
  -v /root/models:/models \
  -p 8000:8000 --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e VLLM_USE_FASTOKENS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=1 \
  vllm/vllm-openai-tuned:latest \
  /models/Swift-Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --gpu-memory-utilization $MEM \
  --max_num_seqs $NUM_SEQS \
  "${QUANT_CONFIG[@]}" \
  --reasoning-parser qwen3  --tool-call-parser qwen3_coder \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.8,"top_p":0.9,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.05}' \
  --trust-remote-code \
  --performance-mode $PERF_MODE \
  --max-num-batched-tokens 16384 --long-prefill-token-threshold 4096 \
  "${MISC_CONFIG[@]}" \
  "${SPEC_CONFIG[@]}" \
  "${OFFLOAD_CONFIG[@]}"
