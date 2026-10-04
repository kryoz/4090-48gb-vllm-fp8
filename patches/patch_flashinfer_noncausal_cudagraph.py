#!/usr/bin/env python3
"""Per-spec non-causal gate for the FlashInfer fa2-native cudagraph support.

Upstream gap (vllm v0.33.0rc0; on v0.32 the whole fa2-branch is absent, so sm89
flashinfer NEVER got UNIFORM_BATCH there): on fa2-native-decode GPUs (pre-sm90:
sm86/sm89) FlashInferMetadataBuilder.get_cudagraph_support downgrades to
UNIFORM_SINGLE_TOKEN_DECODE whenever the GLOBAL attention_config.use_non_causal
is set. DFlash/DSpark set that flag for their non-causal DRAFT block attention,
but the TARGET only ever issues uniform CAUSAL verify queries, so the downgrade
is a false negative: resolve_cudagraph_mode_and_sizes (config/compilation.py,
"CUDAGraphMode... is not supported with spec-decode for attention backend
flashinfer") silently drops FULL decode CUDA graphs to PIECEWISE, and in
PIECEWISE every mamba/GDN layer is a splitting op that runs eager (48 of 64
layers on Qwen3.8-27B) -> single-stream decode loses ~15-35% tok/s.

Fix: judge non-causality from the group's own KV specs. FullAttentionSpec
carries an explicit non_causal flag (merged per-layer in kv_cache_interface),
so a causal target sharing the engine config with a non-causal DFlash drafter
keeps UNIFORM_BATCH and full decode CUDA graphs. Specs WITHOUT an explicit
flag (e.g. DFlash sliding-window draft specs) stay conservative and follow the
global flag, so genuinely non-causal groups still degrade exactly as before.

Anchor verified byte-identical at tag v0.33.0rc0, at 5d35ce3934 (nightly image
pin cu134-nightly-5d35ce3934a7804b298559d2d99eec33792e1eb9), and at e7fa26d5c8.
"""
import os
import py_compile
import sys
import tempfile

TARGET_DEFAULT = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/"
    "flashinfer.py"
)
MARKER = "[club-3090] judge"
ANCHOR = (
    "        kv_specs = iter_layer_specs(kv_cache_spec)\n"
    "        if _use_fa2_native_decode():\n"
    "            # fa2 decodes uniform causal spec queries.\n"
    "            if kv_specs and not vllm_config.attention_config.use_non_causal:\n"
    "                return AttentionCGSupport.UNIFORM_BATCH\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
)
REPLACEMENT = (
    "        kv_specs = iter_layer_specs(kv_cache_spec)\n"
    "        if _use_fa2_native_decode():\n"
    "            # fa2 decodes uniform causal spec queries. [club-3090] judge\n"
    "            # non-causality per spec, not from the global flag: DFlash sets\n"
    "            # the global use_non_causal for its non-causal draft, but the\n"
    "            # causal target only ever issues uniform causal verify queries\n"
    "            # and must keep UNIFORM_BATCH (full decode CUDA graphs). Specs\n"
    "            # without an explicit non_causal flag (e.g. sliding-window\n"
    "            # drafts) stay conservative under the global flag.\n"
    "            use_non_causal = vllm_config.attention_config.use_non_causal\n"
    "            non_causal = False\n"
    "            for spec in kv_specs:\n"
    "                flag = getattr(spec, \"non_causal\", None)\n"
    "                non_causal = non_causal or (\n"
    "                    use_non_causal if flag is None else flag\n"
    "                )\n"
    "            if kv_specs and not non_causal:\n"
    "                return AttentionCGSupport.UNIFORM_BATCH\n"
    "            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
)


def main() -> int:
    target = os.environ.get("VLLM_FLASHINFER_PY", TARGET_DEFAULT)
    try:
        with open(target, encoding="utf-8") as fh:
            src = fh.read()
    except OSError as e:
        print(f"[fi-nc-cg] REFUSE: cannot read target: {e}")
        return 1
    if MARKER in src:
        print("[fi-nc-cg] already patched (idempotent no-op)")
        return 0
    n = src.count(ANCHOR)
    if n != 1:
        print(
            f"[fi-nc-cg] REFUSE: anchor drift - expected exactly 1 fa2-native "
            f"support block, found {n}. Upstream may have reworked "
            f"get_cudagraph_support; re-validate before re-applying."
        )
        return 1
    out = src.replace(ANCHOR, REPLACEMENT)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(out)
        tmp = fh.name
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[fi-nc-cg] REFUSE: patched file does not compile: {e}")
        return 1
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(
        "[fi-nc-cg] applied: fa2-native support gate now judges non-causality "
        "per KV spec (causal target keeps UNIFORM_BATCH under DFlash)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
