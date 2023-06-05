# Image-Captioning-CC3M
This is the repo of image captioning CC3M. I used the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) provided by HuggingFace and the [ViT+GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) to perform image captioning. I froze the vision encoder part of both models to prevent it from being updated, and only trained the language decoder part with a LoRA adapter provided by [PEFT](https://github.com/huggingface/peft). As specified in the instruction, I applied the adapter not only to the attention layer but also to the feed-forward layer.
## Settings
```
docker run --name imgcap -it --gpus all -v /dir:/workspace --ipc host pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
pip install -r requirements.txt
```
## Runs
```
# Using BLIP
python train_blip.py

# Using ViT+GPT2
python train_vit_gpt2.py
```
## Results
- Train/Val loss curve
  - BLIP results
    <img width="1044" alt="스크린샷 2023-06-06 오전 12 26 25" src="https://github.com/sylee0520/Image-Captioning-CC3M/assets/72010172/c79c894d-2257-4915-96ea-b8d27960f789">
- Captioning results
