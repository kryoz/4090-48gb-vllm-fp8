FROM vllm/vllm-openai:latest

COPY tuned_configs/*.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install json_repair fastokens arctic-inference==0.1.1

COPY patches /vllm-workspace/patches

RUN python3 patches/vllm-pr48375-mamba-drop-eagle-block/patch.py
RUN python3 patches/vllm-gdn-mtp-async-spec-order/patch.py
RUN python3 patches/vllm-flashinfer-decode-pin/patch.py

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=180s \
 CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
