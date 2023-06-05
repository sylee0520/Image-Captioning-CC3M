import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
import json
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor, get_scheduler
import wandb
import random
import numpy as np

def captioning(feature_extractor, tokenizer, model, device, path):
    image = Image.open(path)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values)
    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return preds

class ImageCaptioningDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.annotation = json.load(f)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        raw_text = self.annotation[idx]["text"]
        image_path = self.annotation[idx]["image"]
        raw_image = Image.open(image_path)
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert(mode="RGB")
        image = self.processor(images=raw_image, return_tensors="pt").pixel_values.squeeze(0).to('cuda')

        text_inputs = self.tokenizer(text=raw_text, return_tensors="pt", padding="max_length").input_ids[:, :-1].squeeze(0).to('cuda')
        text_labels = self.tokenizer(text=raw_text, return_tensors="pt", padding="max_length").input_ids[:, 1:].squeeze(0).to('cuda')
        
        return image, text_inputs, text_labels
    
    
def main(args):
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    output_path = args.output_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    wandb.login()
    run = wandb.init(
        project="image-captioning",
    )
    train_data = ImageCaptioningDataset(train_data_path)
    val_data = ImageCaptioningDataset(val_data_path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    original_param = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    
    print(f"Trainable parameters of original model: {original_param}")
    
    for param in model.vision_model.parameters():
        param.requires_grad = False
    
    inter = [f'text_decoder.bert.encoder.layer.{i}.intermediate.dense' for i in range(12)]
    out = [f'text_decoder.bert.encoder.layer.{i}.output.dense' for i in range(12)]
    target_modules = ['query', 'value']
    target_modules += inter
    target_modules += out
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules
    )
    model.text_decoder = get_peft_model(model.text_decoder, peft_config)
    
    final_param = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    print(f"Trainable parameters of final model: {final_param}")
    print(f"Ratio of trainable parameters: {final_param/original_param*100}%")
        
    tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    feature_extractor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*epochs)
    
    best_val_loss = 1e10
    for epoch in tqdm(range(epochs)):
        model.train()
        run.log({"epoch": epoch}, commit=True)
        for image, text_inputs, text_labels in tqdm(train_dataloader):
            loss = model(pixel_values=image, input_ids=text_inputs, labels=text_labels).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            run.log({"train_loss": loss}, commit=False)

        if epoch % 1 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for image, text_inputs, text_labels in tqdm(val_dataloader):
                    loss = model(pixel_values=image, input_ids=text_inputs, labels=text_labels).loss
                    val_loss += loss
                    run.log({"val_loss": loss}, commit=False)
                    
            val_loss /= len(val_dataloader)
            
            if val_loss < best_val_loss:
                print(captioning(feature_extractor, tokenizer, model, device, "/workspace/ic/DownloadConceptualCaptions/validation/0.jpg"))
                print(captioning(feature_extractor, tokenizer, model, device, "/workspace/ic/DownloadConceptualCaptions/validation/5.jpg"))
                print(captioning(feature_extractor, tokenizer, model, device, "/workspace/ic/DownloadConceptualCaptions/validation/8.jpg"))
                best_val_loss = val_loss
                model.save_pretrained(output_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path')
    parser.add_argument('--val_data_path')
    parser.add_argument('--output_path')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    
    
    args = parser.parse_args()
    
    main(args)