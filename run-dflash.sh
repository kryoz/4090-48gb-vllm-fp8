#!/bin/bash
set -e

MODEL=Swift-1.5-Qwen3.8-27B-INT8-W8A16-MTP
SPEC_DFLASH='{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":8,"draft_sample_method":"probabilistic"}'
docker run --rm --name vllm --runtime nvidia --gpus all \
  --log-opt max-size=10m --log-opt max-file=3 \
  --cpuset-cpus 1-7 \
  -v /root/.cache/vllm:/root/.cache/vllm \
  -v /root/.triton:/root/.triton \
  -v /root/models:/models \
  -p 8000:8000 --ipc=host \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FASTOKENS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai-tuned:latest \
  /models/${MODEL} \
  --served-model-name qwen3.8-27b \
  --max_num_seqs 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len auto \
  -sc "$SPEC_DFLASH" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --mamba-cache-mode align --enable-mamba-shared-prefix-checkpoint \
  --mamba-ssm-cache-dtype bfloat16 --mamba-block-size 8192 \
  --block-size 32 \
  --attention-backend flashinfer \
  --enable-flashinfer-autotune \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --override-generation-config '{"temperature":0.9,"top_p":0.95,"top_k":20,"min_p":0.0,"seed":7,"presence_penalty":0.0,"repetition_penalty":1.05}' \
  --trust-remote-code \
  --performance-mode interactivity \
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
