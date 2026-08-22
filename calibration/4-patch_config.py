"""
patch_config.py

The problem goes deeper than a single field: the working -FP8 checkpoint uses
a multimodal config class (Qwen3_5Config) with a nested text_config, while the
BF16 calibration source was loaded as a flat Qwen3_5TextConfig. These structures
are incompatible, and merging individual keys (as in the first version of the
script) produces a hybrid configuration that also fails to load.

Solution: take the ENTIRE structure from the working reference config and
overlay only what llm-compressor actually added to the target — the
quantization_config key, which describes the compressed-tensors format for
weights/KV cache.

Everything else from the target (architectural fields) is discarded in favor
of the reference, since those fields describe an incompatible flat structure.

Usage:
    python3 patch_config.py \
        --reference /models/Qwen3.8-27B-FP8/config.json \
        --target /workspace/output/Qwen3.8-27B-FP8-KV-calibrated/config.json
"""

import argparse
import json


# Keys that must come from the target (llm-compressor), not the reference —
# they describe what actually changed as a result of calibration.
KEYS_FROM_TARGET = ["quantization_config"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="config.json of the working checkpoint")
    parser.add_argument("--target", required=True, help="config.json of the calibrated checkpoint")
    parser.add_argument(
        "--backup", action="store_true", default=True,
        help="Save the original target as .bak before overwriting",
    )
    args = parser.parse_args()

    with open(args.reference, "r", encoding="utf-8") as f:
        reference = json.load(f)

    with open(args.target, "r", encoding="utf-8") as f:
        target = json.load(f)

    if args.backup:
        backup_path = args.target + ".bak"
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(target, f, indent=2, ensure_ascii=False)
        print(f"Backup of the original target saved to {backup_path}")

    missing_from_target = [k for k in KEYS_FROM_TARGET if k not in target]
    if missing_from_target:
        print(f"WARNING: expected calibration keys are missing from target: {missing_from_target}")
        print("This is suspicious — verify that oneshot() actually quantized anything.")

    merged = dict(reference)

    for key in KEYS_FROM_TARGET:
        if key in target:
            merged[key] = target[key]
            print(f"Taking {key!r} from target (calibration result)")

    # Architectural consistency check — if the model type/classes differ,
    # replacing the config will not fix the problem. This means the issue is
    # deeper (for example, the wrong source model was downloaded).
    if reference.get("architectures") != target.get("architectures"):
        print(
            f"WARNING: architectures do not match: "
            f"reference={reference.get('architectures')}, target={target.get('architectures')}. "
            f"The merged structure was taken from reference — but make sure that the "
            f"state_dict of the calibrated model actually matches this architecture."
        )

    with open(args.target, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(
        "config.json fully replaced with the reference structure "
        "(+ quantization_config from target)"
    )
    print(f"Written to {args.target}")


if __name__ == "__main__":
    main()
