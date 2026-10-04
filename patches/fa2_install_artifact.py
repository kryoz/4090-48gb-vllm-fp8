"""Install a verified prebuilt FlashAttention plugin; no downloads or CUDA build."""
import argparse
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sysconfig

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()
    if args.tp < 1 or torch.cuda.device_count() < args.tp:
        raise SystemExit(f"FA2 profile needs {args.tp} selected CUDA devices")
    capabilities = {torch.cuda.get_device_capability(i) for i in range(args.tp)}
    if len(capabilities) != 1:
        raise SystemExit(f"FA2 profile requires homogeneous GPU architecture; got {sorted(capabilities)}")
    capability = next(iter(capabilities))
    runtime = Path("/etc/club3090/fa2-runtime.env")
    if capability in ((9, 0), (10, 0)):
        temporary = runtime.with_suffix(".tmp")
        temporary.write_text("# Native FlashAttention FP8 path; no extension required.\n", encoding="utf-8")
        os.replace(temporary, runtime)
        print(f"[fa2] native FlashAttention FP8 path on SM{capability[0]}{capability[1]}")
        return
    if capability not in ((8, 6), (8, 9), (12, 0)):
        raise SystemExit(f"No FP8 FlashAttention kernel path for SM{capability[0]}{capability[1]}")
    if capability != (8, 6):
        print(f"[fa2] WARNING: SM{capability[0]}{capability[1]} is a compile target; physical-GPU validation is not available")
    identity = os.environ.get("FA2_ARTIFACT_ID", "")
    if not re.fullmatch("[0-9a-f]{64}", identity):
        raise SystemExit("FA2_ARTIFACT_ID must identify the pinned kernel artifact")
    artifact = Path("/opt/club3090/fa2-artifacts") / identity
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest.pop("artifact_id")
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if declared != identity or actual != identity or manifest["schema"] != 1:
        raise SystemExit("FA2 artifact manifest identity mismatch")
    abi = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "python_soabi": sysconfig.get_config_var("SOABI"),
           "machine": platform.machine(), "system": platform.system(),
           "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI}
    if abi != manifest["abi"]:
        raise SystemExit(f"FA2 artifact ABI mismatch: runtime={abi}, artifact={manifest['abi']}")
    if f"{capability[0]}.{capability[1]}" not in manifest["compiled_sm"]:
        raise SystemExit(f"FA2 artifact has no compiled kernel for SM{capability[0]}{capability[1]}")
    for name, expected in manifest["files"].items():
        path = artifact / name
        if not path.resolve().is_relative_to(artifact.resolve()):
            raise SystemExit(f"FA2 artifact path escapes its directory: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"FA2 artifact checksum mismatch: {name}")
    # Check the public interface used by the plugin, not unrelated source bytes.
    from vllm.v1.attention.backends import flash_attn
    from vllm.v1.kv_cache_layout import KVCacheLayout
    required = {"num_actual_tokens", "max_query_len", "query_start_loc", "seq_lens", "block_table", "slot_mapping", "causal"}
    if not required.issubset({field.name for field in fields(flash_attn.FlashAttentionMetadata)}):
        raise SystemExit("Unsupported vLLM FlashAttention metadata interface")
    if KVCacheLayout.LBNHC.layer_view_order != (0, 2, 1, 3):
        raise SystemExit("Unsupported vLLM FlashAttention KV layout interface")
    wheels = [artifact / name for name in manifest["files"] if name.endswith(".whl")]
    if len(wheels) != 1:
        raise SystemExit("FA2 artifact must contain exactly one plugin wheel")
    subprocess.run(["python3", "-m", "pip", "install", "--no-deps", "--no-index",
                    "--force-reinstall", str(wheels[0])], check=True)
    exports = {"FA2_FP8KV_LIBRARY": str(artifact / "fa2_fp8kv.so"),
               "FA2_FP8KV_PREFILL_LIBRARY": str(artifact / "fa2_fp8kv_prefill.so")}
    temporary = runtime.with_suffix(".tmp")
    temporary.write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in exports.items()), encoding="utf-8")
    os.replace(temporary, runtime)
    print(f"[fa2] verified artifact={identity}; FLASH_ATTN plugin; runtime vLLM={importlib.metadata.version('vllm')}")


if __name__ == "__main__":
    main()
