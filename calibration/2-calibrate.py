"""
calibrate.py — runs inside the llm-compressor-calib image.

Expected volume mapping (see docker run below):
  /models              -> /root/models on the host (contains the bf16 source model)
  /workspace/dataset    -> calibration_agentic_samples.jsonl (from build_calibration_dataset.py)
  /workspace/output     -> where the calibrated checkpoint will be saved
"""

from datasets import load_dataset
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from llmcompressor import oneshot

# bf16 source model, NOT /models/Qwen3.8-27B-FP8 — KV-cache calibration cannot be
# "added on top" of an already quantized checkpoint; a complete recipe from scratch
# is required.
MODEL_ID = "/models/Qwen3.8-27B"
DATASET_PATH = "/workspace/dataset/calibration_agentic_samples.jsonl"
OUTPUT_DIR = "/workspace/output/Qwen3.8-27B-FP8-KV-calibrated"

NUM_CALIBRATION_SAMPLES = 800
MAX_SEQUENCE_LENGTH = 4096

print(f"Loading {MODEL_ID} (sequential onloading — the entire checkpoint is not loaded into VRAM at once)")
# IMPORTANT: Qwen3.8-27B is a multimodal model (Causal LM + Vision Encoder),
# despite the lack of a -VL suffix in its name. AutoModelForCausalLM matches the
# text-only branch of the mapping (Qwen3_5ForCausalLM) and silently loses the
# vision_config/text_config nesting — which later causes an architecture mismatch
# with the production checkpoint (Qwen3_5ForConditionalGeneration).
# The multimodal auto class must be used, as in the official example on the model card.
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map=None,  # CPU first; oneshot will distribute layers across GPUs
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# llm-compressor reads num_attention_heads/hidden_size/head_dim directly from
# model.config when calculating KV-cache scales, without looking into config.text_config.
# This matters for multimodal wrapper configs (Qwen3_5Config), where these fields
# are nested inside text_config.
# Copy the required attributes to the top level in memory — this does not modify
# the config.json on disk, only the object in the current process, and only affects
# what oneshot() sees during calibration.
if hasattr(model.config, "text_config"):
    text_config = model.config.text_config
    for attr in ("num_attention_heads", "num_key_value_heads", "hidden_size", "head_dim"):
        if hasattr(text_config, attr) and not hasattr(model.config, attr):
            value = getattr(text_config, attr)
            setattr(model.config, attr, value)
            print(f"Promoted {attr}={value} from text_config to the top-level config")

# Verification: does the calibration forward pass actually reach the MTP module?
# If the counter remains at 0 after calibration, activation statistics for MTP
# were not collected, and quantizing its Linear layers (regardless of the ignore
# list) would be blind, equivalent to using q_scale=1.0 as in the original issue.
mtp_call_counts = {}

def _make_hook(name):
    def _hook(module, inputs, outputs):
        mtp_call_counts[name] = mtp_call_counts.get(name, 0) + 1
    return _hook

mtp_hook_handles = []
for name, module in model.named_modules():
    if "mtp" in name.lower() and len(list(module.children())) == 0:  # leaf modules only
        handle = module.register_forward_hook(_make_hook(name))
        mtp_hook_handles.append(handle)

print(f"Registered hooks on {len(mtp_hook_handles)} leaf MTP modules for verification")

print(f"Loading dataset {DATASET_PATH}")
ds = load_dataset("json", data_files=DATASET_PATH, split="train")

def has_user_message(example):
    return any(m.get("role") == "user" for m in example["messages"])

n_before = len(ds)
ds = ds.filter(has_user_message)
n_after = len(ds)
if n_after < n_before:
    print(f"Filtered out {n_before - n_after} sessions without role=user (would break the chat template)")

ds = ds.select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    ignore=[
        "lm_head",
        "model.embed_tokens",
        "re:.*visual.*",
        "re:.*vision.*",
        "re:.*input_layernorm$",
        "re:.*post_attention_layernorm$",
        "re:.*mlp\\.gate$",
        "re:.*mlp\\.shared_expert_gate$",
        "re:.*self_attn\\.k_norm$",
        "re:.*self_attn\\.q_norm$",
        "re:.*linear_attn\\.(A_log|conv1d|dt_bias|in_proj_ba|in_proj_b|in_proj_a|norm)$",
        "re:^mtp\\.(fc|norm|pre_fc_norm_embedding|pre_fc_norm_hidden)$",
        "re:.*mtp.*"
    ],
    config_groups={
        "group_0": {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "dynamic": False,
                "symmetric": True
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "dynamic": True,
                "symmetric": True
            },
            "targets": ["Linear"]
        }
    },
    kv_cache_scheme={
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True
    }
)

print("Starting oneshot calibration...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

for handle in mtp_hook_handles:
    handle.remove()

if not mtp_call_counts:
    print(
        "WARNING: No MTP module was called even once during calibration! "
        "This means the oneshot() forward pass does not reach the MTP path, "
        "so activation statistics for its Linear layers were not collected — "
        "quantization would be blind, equivalent to q_scale=1.0 from the original issue. "
        "In this case, it is better to return to excluding MTP entirely from the recipe "
        "(ignore: re:.*mtp.*) rather than quantizing it 'by eye' according to the vendor's "
        "layer list without actual calibration statistics."
    )
else:
    print(f"MTP modules were called during calibration: {mtp_call_counts}")

print(f"Saving to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
