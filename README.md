# 4090-48gb-vllm-fp8
Tuned FP8 kernel config files for modded RTX 4090 48GB

For docker vllm just build custom image based on `vllm/vllm-openai:latest` at with Dockerfile from the repository:
```bash
docker build . -t vllm/vllm-openai-tuned
```
then use instead of official in your command or service like [run.sh](run.sh)

If you use local vllm install copy `tuned_configs` files to `usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs`
```bash
cp tuned_configs/*.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs
```
