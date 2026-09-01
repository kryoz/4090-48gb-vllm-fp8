#!/bin/sh
docker run --rm --name embeddings \
  --cpuset-cpus 5-7 -e RAYON_NUM_THREADS=3 -e OMP_NUM_THREADS=3 \
  --memory=6g --memory-swap=6g \
  -v /root/models/tei-cache:/data \
  -p 8001:80 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 \
  --model-id=BAAI/bge-m3 \
  --max-batch-tokens=16384 \
  --max-concurrent-requests=64 \
  --auto-truncate
