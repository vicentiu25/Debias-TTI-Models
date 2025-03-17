from transformers import CLIPProcessor, CLIPModel
from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import os
import math
import torch
import gc
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from PIL import Image
import clip

def create_image_dirs(VERSION):
   images = np.load(f'../results/images_eval/images_eval_{VERSION}.npy', allow_pickle=True).item()

   output_dir = f"../results/complete_eval/{VERSION}_images"
   os.makedirs(output_dir, exist_ok=True)

   all_images = []
   captions_mapping = {}

   for prompt, generated_images in images.items():
      for idx, image in enumerate(generated_images):
         image_path = os.path.join(output_dir, f"image_{prompt.replace(' ', '_')}_{idx}.jpg")
         image.save(image_path)
         all_images.append(image_path)
         captions_mapping[image_path] = prompt
   
   return all_images, captions_mapping

def compute_clip_score(VERSION):
   all_images, captions_mapping = create_image_dirs(VERSION)
   clip_score = calculate_clip_score(all_images, captions_mapping=captions_mapping, dataloader_workers = 0)
   print(clip_score)

def read_csv(image_folder):
   all_images = []
   csv_path = 'MS-COCO_val2014_30k_captions.csv'
   df = pd.read_csv(csv_path)

   captions_mapping = {}
   for _, row in df.iterrows():
      image_id = row['image_id']
      caption = row['text']
      image_path = os.path.join(image_folder, f"{image_id}.jpeg")
      if os.path.exists(image_path):
         captions_mapping[image_path] = caption
         all_images.append(image_path)
   return all_images, captions_mapping
      
def calculate_clip_score_for_image(image_path: str, caption: str, clip_model, clip_processor, device):
    """Compute CLIP image-text similarity (cosine similarity between embeddings)."""
    image = Image.open(image_path).convert("RGB")
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    text_inputs = clip_processor(text=[caption], return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)

    # Normalize embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    score = (image_features * text_features).sum().item()
    return score

def compute_clip_individually(folder):
   all_images, captions_mapping = create_image_dirs(VERSION)
   # Model names
   CLIP_MODEL_NAME = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

   # Load models
   model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to("cuda:0").eval()
   preprocess = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

   total_score = 0.0
   num_images = 0

   for image_path in tqdm(all_images):
      caption = captions_mapping[image_path]
      score = calculate_clip_score_for_image(image_path, caption, model, preprocess, "cuda:0")
      total_score += score
      num_images += 1
   final_clip_score = total_score / num_images if num_images > 0 else 0.0
   return final_clip_score

def generate_fid(folder):
   fid, _ = calculate_fid(
      folder,
      get_coco_fid_stats(),
      dataloader_workers = 0
   )
   return fid

# def calculate_coco_fid_index(
#     ModelWrapper: T2IModelWrapper,
#     device: torch.device = 'cuda',
#     seed: Optional[int] = 42,
#     batch_size: int = 1,
#     save_generations_dir: str = 'coco_generations/',
#     start_index: Optional[int] = 0,
#     end_index: Optional[int] = 29999
# ) -> (int, Tuple[dict, dict]):
#     os.makedirs(save_generations_dir, exist_ok=True)
#     # get COCO-30k captions
#     id2caption = get_coco_30k_captions()
#     captions = []
#     ids = []
#     items = list(id2caption.items())

#     # Iterate over the sliced portion of items
#     for d in items[start_index:end_index]:
#        ids.append(d[0])
#        captions.append(d[1])
        
#     # init model
#     model = ModelWrapper(device, save_dir=save_generations_dir, use_saved_images=True, seed=seed)
#     model.set_captions(captions, file_ids=ids)
    
#     # get coco FID stats
#     coco_stats = get_coco_fid_stats()
    
#     return calculate_fid(coco_stats, model, device=device, seed=seed, batch_size=batch_size)

def generate_fid_30k():
   # MS-COCO FID-30k for T2IModelWrapper
   fid, fid_data = calculate_coco_fid(
      StableDiffusion21Wrapper,
      device='cuda:0',
      save_generations_dir='v165_Gen/'
   )

def compute_metrics_model(model):
   image_folder = f'../results/complete_eval/{model}'
   # calculate_coco_fid_index(      
   #    StableDiffusion21Wrapper,
   #    device='cuda:0',
   #    save_generations_dir=image_folder,
   #    end_index = 625)
   print(str(generate_fid(image_folder)) + model)
   print(str(compute_clip_individually(image_folder)) + model)
   
if __name__ == "__main__":
   # print(str(generate_fid(f'../results/complete_eval/sd2.1Gen')) + " sd2.1")
   # print(str(generate_fid(f'../results/complete_eval/v165_Gen')) + " v165")
   
   VERSIONS=["v2_e1_1"]
   for VERSION in VERSIONS:
      print(VERSION)
      print(str(compute_clip_individually(f"../results/complete_eval/{VERSION}_images")) + VERSION)
   #create_image_dirs(VERSION)
   #print(str(compute_clip_individually(f'../results/complete_eval/v165_Gen/')) + " v165")
   # calculate_coco_fid_index(      
   #    StableDiffusion21Wrapper,
   #    device='cuda:0',
   #    save_generations_dir='v165_Gen/',
   #    start_index = 29614)
   #compute_metrics_model('sd2.1Gen')
   # compute_clip_score()
   #generate_fid('../results/images_eval/images_eval_sd2.1Gen.npy')