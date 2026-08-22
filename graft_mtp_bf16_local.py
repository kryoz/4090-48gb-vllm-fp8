#!/usr/bin/env python3
"""
Графт bf16 MTP-головы в Qwen3.8-27B-FP8 (FP8-чекпоинт уже скачан локально).
bf16-исходник тянется штатным hf_hub_download — со всеми ретраями,
докачкой и переключением Xet/LFS-bridge из коробки, без ручных
Range-запросов.

pip install safetensors torch huggingface_hub

Переменные окружения, которые понимает сама huggingface_hub:
  HF_HUB_DISABLE_XET=1        # уйти с Xet на legacy LFS bridge
  HTTPS_PROXY=http://host:port
  HF_HUB_DOWNLOAD_TIMEOUT=120
  HF_TOKEN=...                 # если понадобится

# правим на месте:
python graft_mtp_bf16_hf.py --src /path/to/Qwen3.8-27B-FP8 --inplace

# либо пишем отдельно (непатченные файлы — хардлинком):
python graft_mtp_bf16_hf.py --src /path/to/Qwen3.8-27B-FP8 --out ./Qwen3.8-27B-FP8-mtp-bf16
"""
import argparse, json, os, shutil
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download

BF16_REPO = "Qwen/Qwen3.8-27B"


def link_or_copy(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)  # повторный запуск после сбоя — пересоздаём заново
    try:
        os.link(src_path, dst_path)  # хардлинк — мгновенно, без лишнего места
    except OSError:
        shutil.copy(src_path, dst_path)  # другая ФС — обычная копия


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="локальная папка с Qwen3.8-27B-FP8")
    ap.add_argument("--out", default=None, help="куда писать; без флага правим --src на месте")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    inplace = args.out is None
    out = args.src if inplace else args.out
    if not inplace:
        os.makedirs(out, exist_ok=True)

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    full_index = json.load(open(index_path))
    fp8_map = full_index["weight_map"]

    bf16_index_path = hf_hub_download(BF16_REPO, "model.safetensors.index.json", token=args.token)
    bf16_map = json.load(open(bf16_index_path))["weight_map"]

    mtp_weights = sorted(k for k in fp8_map if k.startswith("mtp.") and k.endswith(".weight"))
    quantized_mtp = [w for w in mtp_weights if f"{w}_scale_inv" in fp8_map]
    print(f"mtp.* тензоров: {len(mtp_weights)}, заквантовано в fp8: {len(quantized_mtp)}")
    if not quantized_mtp:
        print("Заквантованных mtp-линеек нет — чинить нечего.")
        return

    # какие bf16-шарды реально нужны (обычно один, максимум два)
    bf16_shards_needed = sorted({bf16_map[n] for n in quantized_mtp})
    print(f"Качаем {len(bf16_shards_needed)} bf16-шард(а/ов): {bf16_shards_needed}")

    bf16_tensors = {}
    for shard in bf16_shards_needed:
        local_shard = hf_hub_download(BF16_REPO, shard, token=args.token)  # штатный загрузчик HF
        shard_tensors = load_file(local_shard)
        picked = [n for n in quantized_mtp if bf16_map[n] == shard]
        for name in picked:
            bf16_tensors[name] = shard_tensors[name]
        print(f"  из {shard} забрано {len(picked)} тензоров")

    # патчим fp8-шарды локально
    shards_to_patch = sorted({fp8_map[n] for n in quantized_mtp})
    dropped_scale_keys = set()
    for shard in shards_to_patch:
        local_path = os.path.join(args.src, shard)
        tensors = load_file(local_path)
        for name in quantized_mtp:
            if fp8_map[name] != shard:
                continue
            scale_key = f"{name}_scale_inv"
            tensors.pop(scale_key, None)
            dropped_scale_keys.add(scale_key)
            tensors[name] = bf16_tensors[name]

        if inplace:
            backup = local_path + ".orig"
            if not os.path.exists(backup):
                shutil.move(local_path, backup)
            save_file(tensors, local_path, metadata={"format": "pt"})
        else:
            save_file(tensors, os.path.join(out, shard), metadata={"format": "pt"})
        print(f"  пересобран {shard}")

    # config.json и index.json пишутся отдельно ниже — если их тоже хардлинкнуть
    # тут, то запись через open(..., "w") пойдёт в тот же inode и испортит
    # оригинал в --src, т.к. хардлинк это не копия, а второе имя тех же данных
    SKIP_GENERIC_COPY = {"config.json", "model.safetensors.index.json"}
    if not inplace:
        for f in os.listdir(args.src):
            src_path = os.path.join(args.src, f)
            if not os.path.isfile(src_path) or f in shards_to_patch or f in SKIP_GENERIC_COPY:
                continue
            link_or_copy(src_path, os.path.join(out, f))

    # index.json: убираем мёртвые *_scale_inv, пересчитываем total_size
    for k in dropped_scale_keys:
        fp8_map.pop(k, None)
    full_index["weight_map"] = fp8_map
    full_index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(out, s)) for s in set(fp8_map.values())
    )
    json.dump(full_index, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)

    # config.json: mtp-модули в список исключений из квантования
    config_path = os.path.join(out, "config.json")
    if not inplace:
        shutil.copy(os.path.join(args.src, "config.json"), config_path)
    config = json.load(open(config_path))
    qc = config.get("quantization_config", {})
    mtp_modules = sorted({n[: -len(".weight")] for n in quantized_mtp})
    for key in ("modules_to_not_convert", "ignore", "ignored_layers"):
        if key in qc:
            qc[key] = sorted(set(qc[key]) | set(mtp_modules))
            print(f"добавили {len(mtp_modules)} модулей в quantization_config.{key}")
            break
    else:
        qc["modules_to_not_convert"] = mtp_modules
        print("списка исключений не было — создали modules_to_not_convert")
    config["quantization_config"] = qc
    json.dump(config, open(config_path, "w"), indent=2)

    print(f"\nГотово: {out}")
    if inplace:
        print("Оригиналы патченных шардов сохранены как *.safetensors.orig")


if __name__ == "__main__":
    main()
