# Image-Captioning-CC3M
This is the repo of image captioning CC3M. I used the ![BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) provided by HuggingFace and the ![ViT+GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) to perform image captioning. I froze the vision encoder part of both models to prevent it from being updated, and only trained the language decoder part with a LoRA adapter provided by ![PEFT](https://github.com/huggingface/peft). As specified in the instruction, I applied the adapter not only to the attention layer but also to the feed-forward layer.
## Settings
```
docker run --name imgcap -it --gpus all -v /dir:/workspace --ipc host pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
pip install -r requirements.txt
```
## Runs

```

```
