# Image-Captioning-CC3M
This is the repo for [CC3M](https://ai.google.com/research/ConceptualCaptions/download) image captioning. I used the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) provided by HuggingFace and the [ViT+GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) to perform image captioning. I froze the vision encoder part of both models to prevent it from being updated, and only trained the language decoder part with a LoRA adapter provided by [PEFT](https://github.com/huggingface/peft). As specified in the instruction, I applied the adapter not only to the attention layer but also to the feed-forward layer.
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
1. Train/Val loss curve
  - BLIP results
    <img width="1044" alt="스크린샷 2023-06-06 오전 12 26 25" src="https://github.com/sylee0520/Image-Captioning-CC3M/assets/72010172/c79c894d-2257-4915-96ea-b8d27960f789"><br>
  - ViT+GPT2 results
    <img width="1042" alt="스크린샷 2023-06-06 오전 12 28 01" src="https://github.com/sylee0520/Image-Captioning-CC3M/assets/72010172/705a2898-3a7b-4832-9e4f-8e258164cceb">
2. Captioning results
  - BLIP results
    <img width="892" alt="스크린샷 2023-06-06 오전 12 34 13" src="https://github.com/sylee0520/Image-Captioning-CC3M/assets/72010172/afdfab78-eab2-46f9-8dc1-1cecdc044c0f">
  - ViT+GPT2 results
    <img width="878" alt="스크린샷 2023-06-06 오전 12 37 45" src="https://github.com/sylee0520/Image-Captioning-CC3M/assets/72010172/42a0c72b-0b40-415e-80b2-b907926bee64">
