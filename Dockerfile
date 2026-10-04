FROM vllm/vllm-openai:v0.33.0

RUN pip install --no-cache-dir fastokens

# 0.32 -> 0.33: прекомпилированные расширения + rust-фронтенд тянутся с CI-бакета,
# компиляции нет. Deps ставятся ПОЛНОСТЬЮ (без --no-deps): nightly требует
# torch 2.14.0 / flashinfer 0.7.2.post1 / humming-kernels / tokenspeed-mla —
# это и есть целевой стек; pip дотягивает его с PyPI (~5 ГБ).
#COPY vllm-src /src
#RUN VLLM_USE_PRECOMPILED=1 \
#    VLLM_PRECOMPILED_WHEEL_COMMIT=5d35ce3934a7804b298559d2d99eec33792e1eb9 \
#    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
#    SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0 \
#    pip install --no-cache-dir /src

# --- Патчи (после переустановки vllm применяются заново; идемпотентны,
# anchor-checked: при дрейфе сборка ПАДАЕТ, а не служит непатченной) ---

# vllm#48375 (открыт): MambaManager игнорирует drop_eagle_block — MTP × prefix-caching
# портит recurrent-state KV (~20% acceptance, bug vllm#43559). Анкор жив на 5d35ce3934.
COPY patches/patch_mamba_drop_eagle_block.py /tmp/pr48375.py
RUN python3 /tmp/pr48375.py && rm -f /tmp/pr48375.py

# vllm#51581 (открыт): DFlash2 W4A16-drafter — деквантиз qkv_proj перед fused-KV
# срезом. Нужен только для method=dflash; для MTP инертен.
COPY patches/dflash_dense_kv.py /tmp/dflash51581.py
RUN python3 /tmp/dflash51581.py && rm -f /tmp/dflash51581.py

# vllm#40756: FlashInfer decode workspace unpin — Xid 31 при MTP c>=4. No-op без flashinfer.
COPY patches/patch_flashinfer_decode_pin.py /tmp/fi_pin.py
RUN python3 /tmp/fi_pin.py && rm -f /tmp/fi_pin.py

# НОВОЕ (2026-10-27): per-spec non-causal gate в get_cudagraph_support —
# causal-таргет при dflash держит UNIFORM_BATCH → FULL decode CUDA-графы
# (в PIECEWISE все 48 GDN-слоев eager). Главный рычаг single-stream decode.
COPY patches/patch_flashinfer_noncausal_cudagraph.py /tmp/fi_nc_cg.py
RUN python3 /tmp/fi_nc_cg.py && rm -f /tmp/fi_nc_cg.py

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=180s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
