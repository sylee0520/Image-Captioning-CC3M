# Image-Captioning-CC3M
This is the repo for CC3M image captioning. I used the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) provided by HuggingFace and the [ViT+GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) to perform image captioning. I froze the vision encoder part of both models to prevent it from being updated, and only trained the language decoder part with a LoRA adapter provided by [PEFT](https://github.com/huggingface/peft). As specified in the instruction, I applied the adapter not only to the attention layer but also to the feed-forward layer.
## Settings
- Environment setting
```
git clone https://github.com/sylee0520/Image-Captioning-CC3M.git
docker run --name imgcap -it --gpus all -v /Image-Captioning-CC3M:/workspace --ipc host pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
pip install -r requirements.txt
```
- Dataset setting
Please convert an original annotation file to following format.
```
[
  {
    "text": "there is a new bespectacled post up on my blog !",
    "image": "/workspace/data/validation/3808.jpg"
  },
  {
    "text": "colorful illustration of girl in the kitchen .",
    "image": "/workspace/ic/DownloadConceptualCaptions/validation/6208.jpg"
  },
  ...
]
```
## Runs
```
# Using BLIP
python train_blip.py \
--train_data_path <train-data-path> \
--val_data_path <val-data-path> \
--output_path <output-path> \
--epochs 100 \
--batch_size 32 \
--lr 1e-5 \
--seed 42

# Using ViT+GPT2
python train_vit_gpt2.py
--train_data_path <train-data-path> \
--val_data_path <val-data-path> \
--output_path <output-path> \
--epochs 100 \
--batch_size 16 \
--lr 1e-5 \
--seed 42
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
