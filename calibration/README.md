1. Build the initial dataset from omp.sh logs (pi.dev can also be used, but the script will need some minor adaptation).

2. Prepare the dataset, most likely on your local machine where you worked with the agent:
   1. `python 1-prepare_dataset.py`

3. Copy the resulting `calibration_agentic_samples.jsonl` dataset to the server where you normally run inference, into `/root/calibration/dataset`:

4. Use the full-size BF16 Qwen3.8-27B model.

5. Build calibration image:
   ```bash
   docker build -t llm-compressor-calib:latest -f Dockerfile.calib .
   ```

6. Run calibration (pay attention to mount paths from your host machine):
   ```bash
   docker run --rm --runtime nvidia --gpus all --ipc=host \
     -v /root/models:/models \
     -v ./:/workspace \
     llm-compressor-calib:latest \
     2-calibrate.py
   ```

7. Transplant the MTP layer:
   ```bash
   docker run --rm --runtime nvidia --gpus all --ipc=host \
      -v /root/models:/models \
      -v /root/calibration:/workspace \
      llm-compressor-calib:latest \
      3-transplant_mtp.py
  ```

8. Patch the configuration (no docker needed here):
   ```bash
   python3 4-patch_config.py \
    --reference /root/models/Qwen3.8-27B-FP8/config.json \
    --target /root/calibration/output/Qwen3.8-27B-FP8-KV-calibrated/config.json

   python3 5-fix_quantization_ignore.py \
     --config /root/calibration/output/Qwen3.8-27B-FP8-KV-calibrated/config.json
   ```
