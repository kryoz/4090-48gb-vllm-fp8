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
then use it instead of official in your command or service like [run-mtp.sh](run-mtp.sh)

There's also DFash2 version which requires additional drafter model (and memory too) [run-dflash.sh](run-dflash.sh)


