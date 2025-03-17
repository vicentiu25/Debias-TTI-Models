import os
import json
import torch
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import timm
from torch.nn.functional import cosine_similarity
import numpy as np
import clip

# ===========================
# CONFIGURATION
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model names
CLIP_MODEL_NAME = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
DINO_MODEL_NAME = "dinov2_vitg14"

# Load models
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# Preprocessing for DINO
dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# ===========================
# FUNCTIONS
# ===========================
def compute_clip_score(prompt, image_path):
    """Compute CLIP image-text similarity (cosine similarity between embeddings)."""
    image = Image.open(image_path).convert("RGB")
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    text_inputs = clip_processor(text=[prompt], return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)

    # Normalize embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    score = (image_features * text_features).sum().item()
    return score


def compute_clip_image_similarity(image_path1, image_path2):
    """Compute CLIP image-image similarity correctly."""
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    # Process each image separately
    image1_inputs = clip_processor(images=image1, return_tensors="pt").to(device)
    image2_inputs = clip_processor(images=image2, return_tensors="pt").to(device)

    with torch.no_grad():
        image1_features = clip_model.get_image_features(**image1_inputs)
        image2_features = clip_model.get_image_features(**image2_inputs)

    # Normalize features
    image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
    image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = cosine_similarity(image1_features, image2_features).item()
    return similarity


def compute_dino_features(image_path):
    """Extract DINOv2 features for an image."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = dino_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dino_model(input_tensor)
    normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
    return normalized_features


def compute_dino_similarity(image_path1, image_path2):
    """Compute cosine similarity between two images using DINOv2."""
    features1 = compute_dino_features(image_path1)
    features2 = compute_dino_features(image_path2)
    similarity = cosine_similarity(features1, features2).item()
    return similarity


def compute_scores_per_prompt(prompt, img_paths, original_img_paths):
    """Compute scores for all images of a single prompt."""
    clip_text_scores = []
    clip_image_scores = []
    dino_similarities = []

    for gen_path, orig_path in zip(img_paths, original_img_paths):
        clip_text_scores.append(compute_clip_score(prompt, gen_path))
        clip_image_scores.append(compute_clip_image_similarity(gen_path, orig_path))
        
    avg_clip_t = np.mean(clip_text_scores)
    std_clip_t = np.std(clip_text_scores)
    avg_clip_i = np.mean(clip_image_scores)
    std_clip_i = np.std(clip_image_scores)

    return avg_clip_t, std_clip_t, avg_clip_i, std_clip_i


# ===========================
# MAIN PROCESSING LOOP
# ===========================
if __name__ == "__main__":
    prompts_path = './data/1-prompts/occupation.json'
    save_dir = './sd1.5_ours/generated-images/test_prompts_occupation'
    original_images_dir = './sd1.5/generated-images/test_prompts_occupation'
    num_imgs_per_prompt = 160
    
    with open(prompts_path, 'r') as f:
        experiment_data = json.load(f)

    test_prompts = experiment_data["test_prompts"]
    overall_clip_t_scores = []
    overall_clip_i_scores = []

    for i, prompt_i in tqdm(enumerate(test_prompts), total=len(test_prompts), desc='Prompts', leave=True):
        save_dir_prompt_i = os.path.join(save_dir, f"prompt_{i}")
        original_dir_prompt_i = os.path.join(original_images_dir, f"prompt_{i}")

        img_paths = [os.path.join(save_dir_prompt_i, f"img_{j}.jpg") for j in range(num_imgs_per_prompt)]
        original_img_paths = [os.path.join(original_dir_prompt_i, f"img_{j}.jpg") for j in range(num_imgs_per_prompt)]

        avg_clip_t, std_clip_t, avg_clip_i, std_clip_i = compute_scores_per_prompt(prompt_i, img_paths, original_img_paths)

        print(f"\nPrompt {i} Results:")
        print(f"  Avg CLIP-T: {avg_clip_t:.4f}, Std CLIP-T: {std_clip_t:.4f}")
        print(f"  Avg CLIP-I: {avg_clip_i:.4f}, Std CLIP-I: {std_clip_i:.4f}")

        overall_clip_t_scores.append(avg_clip_t)
        overall_clip_i_scores.append(avg_clip_i)

    overall_avg_clip_t = np.mean(overall_clip_t_scores)
    overall_std_clip_t = np.std(overall_clip_t_scores)
    overall_avg_clip_i = np.mean(overall_clip_i_scores)
    overall_std_clip_i = np.std(overall_clip_i_scores)

    print("\nOverall Results:")
    print(f"CLIP-T: {overall_avg_clip_t:.4f} ± {overall_std_clip_t:.4f}")
    print(f"CLIP-I: {overall_avg_clip_i:.4f} ± {overall_std_clip_i:.4f}")
