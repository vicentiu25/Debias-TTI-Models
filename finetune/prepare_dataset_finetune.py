import numpy as np
from PIL import Image
from datasets import Dataset
from diffusers import DiffusionPipeline
import jsonlines
import pandas as pd
import torch
import random
from openai import OpenAI
#from deepface import DeepFace

data = np.load("image_data.npy", allow_pickle=True)

def opposite_gender(gender:str) -> str:
    if gender.__eq__("Woman"):
        return "Man"
    else:
        return "Woman"

def opposite_race(race:str) -> str:
    races = ["white", "black", "asian", "middle eastern", "latino hispanic", "indian"]
    races.remove(race)
    return random.choice(races)

def compute_caption_pair(anchor, gender, race):
    prompt = """ 
    Example:
    Initial caption: shoveling snow, by hand
    Extracted attributes : asian woman
    Re-write: %W

    Task:
    Initial caption: %A
    Extracted attributes: %R %M
    Re-write:"""

    generation_text = "A colored photo of an asian woman shoveling snow, by hand, with her face clearly visible."
    rewritten_text = "An asian woman shoveling snow, by hand."
    
    generation_prompt = prompt.replace("%W", generation_text)
    generation_prompt = generation_prompt.replace("%A", anchor)
    generation_prompt = generation_prompt.replace("%R", race)
    generation_prompt = generation_prompt.replace("%M", gender)
    
    rewritten_prompt = prompt.replace("%W", rewritten_text)
    rewritten_prompt = rewritten_prompt.replace("%A", anchor)
    rewritten_prompt = rewritten_prompt.replace("%R", race)
    rewritten_prompt = rewritten_prompt.replace("%M", gender)

    generation_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your goal is to help me rewrite some automatically generated captions for an image-generation task. It is important to preserve the race/gender attributes in these captions, but to make them grammatically correct. Additionally, prioritize generating images where the face of the depicted individual is clearly visible."},
            {"role": "user", "content": generation_prompt}
        ]
    )

    rewritten_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your goal is to help me rewrite some automatically generated captions for an image-generation task. It is important to preserve the race/gender attributes in these captions, but to make them grammatically correct."},
            {"role": "user", "content": rewritten_prompt}
        ]
    )

    generation_caption = generation_completion.choices[0].message.content
    rewritten_caption = rewritten_completion.choices[0].message.content
    return generation_caption, rewritten_caption

def generate_image_pair(text, gender, race):
    #while True:
    image = pipe(prompt=text).images[0]
        #prediction = DeepFace.analyze(np.array(image), actions=['gender', 'race'], detector_backend = 'mtcnn', enforce_detection = False)
        #if(len(prediction) == 1 and pred[0]['dominant_race'] == race and predictions[0]['dominant_gender'] == gender):
    return image


def preprocess_entry(entry, idx, writer, negative_writer, positive_writer):
    image_path = f"../datasets/anchor_dataset/train/{idx}.jpg"
    positive_image_path = f"../datasets/positive_dataset/train/{idx}_positive.jpg"
    negative_image_path = f"../datasets/negative_dataset/train/{idx}_negative.jpg"

    positive_gender = opposite_gender(entry["gender"])
    positive_race = opposite_race(entry["race"])

    positive_generation_text, positive_rewritten_text = compute_caption_pair(entry["anchor"], positive_gender, positive_race)
    negative_generation_text, negative_rewritten_text = compute_caption_pair(entry["anchor"], entry["gender"], entry["race"])

    positive_image = generate_image_pair(positive_generation_text, positive_gender, positive_race)
    negative_image = generate_image_pair(negative_generation_text, entry["gender"], entry["race"])

    entry["image"].save(image_path) 
    positive_image.save(positive_image_path)
    negative_image.save(negative_image_path) 

    writer.write( {
        "file_name": f"{idx}.jpg",
        "text": entry["anchor"]
    } )
    positive_writer.write( {
        "file_name": f"{idx}_positive.jpg",
        "positive_text": positive_rewritten_text
    } )
    negative_writer.write( {
        "file_name": f"{idx}_negative.jpg",
        "negative_text": negative_rewritten_text
    } )

#stabilityai/stable-diffusion-xl-base-1.0
#stabilityai/stable-diffusion-2-1
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

client = OpenAI(api_key='')

def generate_images(data, start_index, end_index):
    with jsonlines.open(f"../datasets/anchor_dataset/train/metadata{end_index}.jsonl", "w") as writer, \
        jsonlines.open(f"../datasets/positive_dataset/train/metadata{end_index}.jsonl", "w") as positive_writer, \
        jsonlines.open(f"../datasets/negative_dataset/train/metadata{end_index}.jsonl", "w") as negative_writer:

        for idx in range(start_index - 1, min(end_index, len(data))):
            preprocess_entry(data[idx], idx + 1, writer, negative_writer, positive_writer)

generate_images(data, 3201, 4200)