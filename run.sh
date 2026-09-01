#!/bin/bash

CTX_SIZE=auto
QUANT="fp8_e4m3"
QUANT_CONFIG=(--dtype bfloat16 --kv-cache-dtype $QUANT --max-model-len $CTX_SIZE --kv-cache-dtype-skip-layers sliding_window)

# include the option into QUANT_CONFIG if your model NOT calibrated for fp8 kv cache
# --kv-cache-dtype-skip-layers sliding_window

PERF_MODE="balanced"
MEM=0.95
NUM_SEQS=4

SPEC_MTP='{"method": "mtp", "num_speculative_tokens": 3}' 
SPEC_CONFIG=(-sc "$SPEC_MTP") 
MISC_CONFIG=(--async-scheduling --enable-prefix-caching --enable-auto-tool-choice --enable-chunked-prefill --mamba-cache-mode align --prefix-match-unit 16 --block-size 32)

docker run --rm --name vllm --runtime nvidia --gpus all --shm-size 16gb \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  --cpuset-cpus 1-3 \
  -v /root/.cache/vllm:/root/.cache/vllm \
  -v /root/.triton:/root/.triton \
  -v /root/models:/models \
  -p 8000:8000 --ipc=host \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  -e VLLM_USE_FASTOKENS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=1 \
  vllm/vllm-openai-tuned:latest \
  /models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --gpu-memory-utilization $MEM \
  --max_num_seqs $NUM_SEQS \
  "${QUANT_CONFIG[@]}" \
  --reasoning-parser qwen3  --tool-call-parser qwen3_coder \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.8,"top_p":0.9,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"preserve_thinking": true,"reasoning_effort": "medium"}' \
  --trust-remote-code \
  --performance-mode $PERF_MODE \
  --max-num-batched-tokens 8192 \
  --attention-backend flashinfer --enable-flashinfer-autotune \
  "${MISC_CONFIG[@]}" \
  "${SPEC_CONFIG[@]}"
