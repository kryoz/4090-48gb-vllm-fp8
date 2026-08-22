"""
transplant_mtp.py

Transplants MTP weights from a working reference checkpoint (Qwen3.8-27B-FP8)
into our calibrated checkpoint, DEQUANTIZING them back to bf16 rather than
copying the raw FP8 bytes as-is.

The reference uses a block-wise FP8 scheme (weight_block_size: [128, 128],
DeepSeek-V3-style), while our final quantization_config describes these MTP
modules as NOT quantized (via the ignore list). Therefore, the loader expects
plain bf16 tensors here rather than block-wise FP8 tensors.

Dequantization:
    weight_bf16[i,j] = weight_fp8[i,j].float() * scale[i//128, j//128]

MTP norm/gate tensors (which are not quantized in the reference) are copied
as-is without dequantization — they simply do not have a corresponding
scale tensor.
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REFERENCE_DIR = Path("/models/Qwen3.8-27B-FP8")
TARGET_DIR = Path("/workspace/output/Qwen3.8-27B-FP8-KV-calibrated")
BLOCK_SIZE = 128  # from the reference checkpoint's weight_block_size


def find_scale_key(weight_key: str, weight_map: dict) -> str | None:
    candidates = [
        weight_key + "_scale_inv",
        weight_key + "_scale",
        weight_key.replace(".weight", ".weight_scale_inv"),
        weight_key.replace(".weight", ".weight_scale"),
    ]

    for c in candidates:
        if c in weight_map:
            return c

    return None


def main():
    with open(REFERENCE_DIR / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        ref_index = json.load(f)

    ref_weight_map = ref_index["weight_map"]

    mtp_keys = sorted(k for k in ref_weight_map if k.startswith("mtp."))
    scale_key_set = {k for k in mtp_keys if k.endswith(("_scale_inv", "_scale"))}
    base_weight_keys = [k for k in mtp_keys if k not in scale_key_set]

    print(f"Found {len(base_weight_keys)} base mtp.* tensors in the reference checkpoint")

    # Group by shard file so that each file only needs to be opened once.
    shard_to_keys = defaultdict(list)
    needed_keys = set(base_weight_keys)

    for k in base_weight_keys:
        scale_key = find_scale_key(k, ref_weight_map)
        if scale_key:
            needed_keys.add(scale_key)

    for k in needed_keys:
        shard_to_keys[ref_weight_map[k]].append(k)

    raw_tensors = {}

    for shard_file, keys in shard_to_keys.items():
        path = REFERENCE_DIR / shard_file

        with safe_open(str(path), framework="pt") as f:
            for k in keys:
                raw_tensors[k] = f.get_tensor(k)

    output_tensors = {}
    n_dequantized = 0
    n_copied_plain = 0

    for weight_key in base_weight_keys:
        weight_tensor = raw_tensors[weight_key]
        scale_key = find_scale_key(weight_key, ref_weight_map)
        if scale_key is not None:
            scale_tensor = raw_tensors[scale_key]
            output_tensors[weight_key] = weight_tensor

            target_scale_key = (
                scale_key[: -len("_inv")] if scale_key.endswith("_inv") else scale_key
            )
            output_tensors[target_scale_key] = scale_tensor

            n_dequantized += 1
            print(
                f"  copied FP8 as-is: {weight_key}  {tuple(weight_tensor.shape)}  "
                f"scale: {scale_key} -> {target_scale_key}"
            )
        else:
            # Non-quanted tensor (norm/gate) copy as is
            output_tensors[weight_key] = weight_tensor.to(torch.bfloat16)
            n_copied_plain += 1

    print(
        f"\nTotal: dequantized {n_dequantized}, "
        f"copied as-is {n_copied_plain}"
    )

    new_shard_name = "mtp.safetensors"

    save_file(
        output_tensors,
        str(TARGET_DIR / new_shard_name),
        metadata={"format": "pt"},
    )

    print(f"New shard with MTP tensors written: {new_shard_name}")

    target_index_path = TARGET_DIR / "model.safetensors.index.json"
    single_file_path = TARGET_DIR / "model.safetensors"

    if target_index_path.exists():
        # Already-sharded checkpoint — normal path.
        with open(target_index_path, "r", encoding="utf-8") as f:
            tgt_index = json.load(f)

        tgt_weight_map = tgt_index["weight_map"]

        old_mtp_keys_in_target = [
            k for k in tgt_weight_map
            if k.startswith("mtp.")
        ]

        print(
            f"Target checkpoint already contains "
            f"{len(old_mtp_keys_in_target)} mtp.* keys — overriding them"
        )

        backup_path = target_index_path.with_suffix(".json.bak")
        shutil.copy(target_index_path, backup_path)

        print(f"Index backup saved: {backup_path}")

    elif single_file_path.exists():
        # save_pretrained() did not shard the checkpoint — the entire checkpoint
        # is stored in a single model.safetensors file, so no index.json was
        # created.
        #
        # Rebuild the index by enumerating all existing keys directly through
        # safetensors (without loading tensor data into memory — only the header
        # is accessed), then add our new shard.
        print(
            "index.json not found — the original checkpoint was not sharded. "
            "Rebuilding the index."
        )

        tgt_weight_map = {}

        with safe_open(str(single_file_path), framework="pt") as f:
            for key in f.keys():
                tgt_weight_map[key] = "model.safetensors"

        # total_size is mostly informational for loaders.
        # Calculate it from the actual file sizes on disk.
        total_size = (
            single_file_path.stat().st_size
            + (TARGET_DIR / new_shard_name).stat().st_size
        )

        old_mtp_keys_in_target = [
            k for k in tgt_weight_map
            if k.startswith("mtp.")
        ]

        print(
            f"Single-file checkpoint contained "
            f"{len(old_mtp_keys_in_target)} mtp.* keys — overriding them"
        )

        tgt_index = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": tgt_weight_map,
        }

    else:
        raise FileNotFoundError(
            f"Neither {target_index_path} nor {single_file_path} was found — "
            f"check that calibration was actually saved to {TARGET_DIR}"
        )

    for k in output_tensors:
        tgt_weight_map[k] = new_shard_name

    tgt_index["weight_map"] = tgt_weight_map

    with open(target_index_path, "w", encoding="utf-8") as f:
        json.dump(tgt_index, f, indent=2)

    print(f"\nDone. index.json written/updated: {target_index_path}")
    print(f"MTP tensors transplanted to {TARGET_DIR / new_shard_name}")
    print("IMPORTANT: make sure MTP is still listed as excluded (not quantized)")
    print("in quantization_config.ignore in your config.json — these tensors are now plain bf16.")


if __name__ == "__main__":
    main()
