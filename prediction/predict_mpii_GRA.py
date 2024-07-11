from PIL import Image, UnidentifiedImageError
import requests
import numpy as np
from deepface import DeepFace
from torch.utils.data import Dataset
from typing import Optional
import os

class MpiiGRAPredictor:
    """A race predictor for images with faces."""

    def __init__(self,
                 dataset: Optional[Dataset]):
        """Initialize the predictor."""
        self.dataset = dataset

    def compute_age_group(self, age: int) -> str:
        if age < 18:
            return "child"
        if age < 35:
            return "young"
        if age < 60:
            return "middle-aged"
        return "senior"  

    def __call__(self):
        """Identifies the images with one face and predicts the gender, race and age of the person."""
        gra_predicted = []
        for idx in range(len(self.dataset.text)):
            try:
                image_path = self.dataset.image[idx]
                absolute_image_path = os.path.abspath(image_path)
                image = Image.open(absolute_image_path)
                img_array = (np.array(image))

                ## DeepFace.analyze finds all faces in the image and predicts the race of each one
                predictions = DeepFace.analyze(img_array, actions=['age', 'gender', 'race'])
                #ratio_face_image = predictions[0]['region']['w'] * predictions[0]['region']['h'] / (image.height * image.width) * 100
                ## We only want images with one face, and the face should cover between 2% and 33% of the image
                if(len(predictions) == 1 and len(self.dataset.text[idx]) > 0):
                    gra_predicted.append({
                        "text": self.dataset.text[idx],
                        "image": image,
                        "gender": predictions[0]['dominant_gender'],
                        "race": predictions[0]['dominant_race'],
                        "age": self.compute_age_group(predictions[0]['age'])
                    })
                if len(gra_predicted)%10 == 0:
                    print(str(len(gra_predicted)) + " / " + str(idx))
                #if len(gra_predicted) == 20:
                    #break
            ## UnidentifiedImageError is raised in case the url is no longer available
            ## ValueError is raised in case the analyze function did not find any face
            ## ReadTimeout is raised in case the get operation to read the image timed out
            except (UnidentifiedImageError, requests.ReadTimeout, requests.ConnectionError, ValueError, Exception) as exception:
                print(str(exception))
        
        return gra_predicted
