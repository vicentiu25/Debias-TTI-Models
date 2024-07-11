from diffusers import DiffusionPipeline
import torch
import numpy as np
from numpy import save, asarray
import matplotlib.pyplot as plt

class ImageGeneration:
    """."""

    def __init__(self,
                prompts: list[str],
                version: str):
        """."""
        self.prompts = prompts
        self.VERSION = version

    def plot_images(self, outputs, few):
        plt.figure(figsize=(30, 30))
        plotted_images = 0
        plt.suptitle(f"images {self.VERSION}", fontsize=50)
        for i, (prompt, images) in enumerate(outputs.items()):
            num_images = min(len(images), 5)
            for j in range(num_images):
                ax = plt.subplot(len(outputs), num_images, plotted_images + j + 1)
                plt.imshow(images[j])
                if j==0:
                    plt.title(prompt, fontsize=20)
                plt.axis("off")
            plotted_images += num_images
        if few == True:
            plt.savefig(f'few_generated_images_{self.VERSION}.jpg', format='jpg')
        else:
            plt.savefig(f'all_generated_images_{self.VERSION}.jpg', format='jpg')

    def generate_images(self) -> str:
        #finetune/outputSD2_v1
        #stabilityai/stable-diffusion-2-1
        pipe = DiffusionPipeline.from_pretrained(f"finetune/outputSD2_{self.VERSION}", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to("cuda")

        images_to_generate = 25
        outputs = {}

        for i, prompt in enumerate(self.prompts):
            generated_images = []
            for _ in range(images_to_generate):
                image = pipe(prompt=prompt).images[0]
                generated_images.append(image)
            outputs.update({prompt: generated_images})
            if i == 4:
                self.plot_images(outputs, True)
        self.plot_images(outputs, False)

        return outputs

    def __call__(self):
        """."""
        images = self.generate_images()

        return images

#if __name__ == "__main__":
    #prompt_generator = PromptGeneration()
    #prompts = prompt_generator()
    #image_generator = ImageGeneration(prompts = prompts)
    #images = image_generator()
    #save(f"images_eval_{VERSION}.npy", images)