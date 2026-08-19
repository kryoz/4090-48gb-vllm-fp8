FROM vllm/vllm-openai:latest

COPY tuned_configs/*.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install json_repair fastokens arctic-inference==0.1.1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
 CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
