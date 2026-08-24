# Tools for tuning vllm with modded RTX 4090 48gb and FP8

You can find here:
- tuned FP8 kernel config files for modded RTX 4090 48GB;
- optimized vllm run script;
- kv cache calibration workflow with your own real agentic dataset.

## Setup

Official vllm docs recommend running it from docker.
It's good to easy reproduct env but modding can be found a bit complicated for someone.
Actually we're going to build a build custom image based on `vllm/vllm-openai:latest`. 
Grab [Dockerfile](Dockerfile) and build modded image:

```bash
docker build . -t vllm/vllm-openai-tuned
```
then use it instead of official in your command or service like [run.sh](run.sh)

If you use local vllm install copy `tuned_configs` files to `usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs`
```bash
cp tuned_configs/*.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs
```

## Improved chat template

It's recommended to use [froggeric's tuned chat template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) to avoid thinking loops and other bugs.

I included [it](chat_template.jinja) with default reasoning set to `medium`. 

To use it just copy it into your models directory to overwrite the default one.
