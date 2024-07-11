from deepface import DeepFace
import numpy as np

class PredictionsGenerator:
    def __init__(self,
                images):
        """."""
        self.images = images
    
    def __call__(self):
        """."""
        predicted_imgs = {}
        for prompt, generated_images in self.images.items():
            prompt_predictions = []
            for i in range(len(generated_images)):
                try:
                    img_array = np.array(generated_images[i])
                    pred = DeepFace.analyze(img_array, actions=['gender', 'race'], detector_backend='retinaface')
                    prompt_predictions.append({
                        'image' : img_array,
                        'race' : pred[0]['dominant_race'],
                        'gender' : pred[0]['dominant_gender']
                    })
                except ValueError:
                    pass
            predicted_imgs[prompt] = prompt_predictions
        return predicted_imgs