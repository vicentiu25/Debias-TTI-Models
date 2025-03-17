import torch
from PIL import Image
from T2IBenchmark import T2IModelWrapper
from diffusers import DiffusionPipeline

class StableDiffusion21Wrapper(T2IModelWrapper):
    def load_model(self, device: torch.device):
        """Initialize model here"""
        self.pipe = DiffusionPipeline.from_pretrained(f"finetune/outputSD2_v165_both", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        
    def generate(self, caption: str) -> Image.Image:
        """Generate PIL image for provided caption"""
        self.pipe.to("cuda")
        return self.pipe(prompt=caption).images[0]