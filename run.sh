#!/bin/bash

SPEC_MTP='{"method": "mtp", "num_speculative_tokens": 3}' 
SPEC_CONFIG=(-sc "$SPEC_MTP") 
ASYNC_SCHED=(--async-scheduling --enable-prefix-caching --mamba-cache-mode align)

CTX_SIZE=auto
QUANT="fp8_e4m3"
QUANT_CONFIG=(--kv-cache-dtype "$QUANT" --kv-cache-dtype-skip-layers sliding_window --max-model-len $CTX_SIZE)

PERF_MODE="interactivity"
BATCH_SIZE=8192

MEM=0.96
NUM_SEQS=3

docker run --rm --name vllm --runtime nvidia --gpus all \
  --cpuset-cpus 1-3 \
  -v /root/.cache/vllm:/root/.cache/vllm \
  -v /root/.triton:/root/.triton \
  -v /root/models:/models \
  -p 8000:8000 --ipc=host \
  -e OMP_NUM_THREADS=1 \
  -e VLLM_USE_FASTOKENS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai-tuned:latest \
  /models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --gpu-memory-utilization $MEM \
  --max_num_seqs $NUM_SEQS \
  --api-server-count 3 \
  "${QUANT_CONFIG[@]}" \
  --block-size 32 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --language-model-only --enable-chunked-prefill --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.8,"top_p":0.9,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs '{"preserve_thinking": true,"reasoning_effort": "low"}' \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --attention-backend flashinfer \
  --performance-mode $PERF_MODE \
  --max-num-batched-tokens $BATCH_SIZE \
  "${ASYNC_SCHED[@]}" \
  "${SPEC_CONFIG[@]}"
