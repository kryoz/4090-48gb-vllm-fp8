#!/bin/bash
set -e

MODEL=Swift-1.5-Qwen3.8-27B-INT8-W8A16-MTP

QUANT_CONFIG=(--dtype bfloat16 --kv-cache-dtype fp8_e4m3 --max-model-len auto)

PERF_MODE="interactivity"
NUM_SEQS=2

SPEC_MTP='{"method": "mtp", "num_speculative_tokens": 4, "draft_lm_head_quantization": "nvfp4"}'
SPEC_CONFIG=(-sc "$SPEC_MTP")

MODEL_PARAMS='{"temperature":0.9,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.05,"seed":7}'

MISC_CONFIG=(
  --async-scheduling
  --enable-prefix-caching
  --enable-auto-tool-choice
  --enable-chunked-prefill
  --mamba-cache-mode align
  --enable-mamba-shared-prefix-checkpoint
  --mamba-block-size 8192
  --mamba-ssm-cache-dtype bfloat16
  --block-size 32
  --attention-backend flashinfer
  --enable-flashinfer-autotune
  --gpu-memory-utilization 0.95
  --override-generation-config "$MODEL_PARAMS"
)

docker run --rm --name vllm --runtime nvidia --gpus all \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
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
  --max_num_seqs $NUM_SEQS \
  "${QUANT_CONFIG[@]}" \
  --reasoning-parser qwen3  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --performance-mode $PERF_MODE \
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096 \
  "${MISC_CONFIG[@]}" \
  "${SPEC_CONFIG[@]}"
