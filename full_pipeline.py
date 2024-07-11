from finetune.test_text_to_image import TextToImage, TextToImageSDXL
from evaluation.generate_prompts import PromptGeneration
from evaluation.generate_images import ImageGeneration
import unittest
import logging
import sys
import logging.config
import json
from numpy import save, asarray
import numpy as np

FORMATTER = logging.Formatter('"%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"')

logger = logging.getLogger(__name__) 
sys.stdout.flush()
console_handler = logging.StreamHandler()
console_handler.setFormatter(FORMATTER)
logger.addHandler(console_handler) 
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    VERSION = "v167_both3"
    unittest.main(argv=[''], defaultTest='TextToImage.test_text_to_image', exit = False)

    prompt_generator = PromptGeneration()
    prompts = prompt_generator()
    image_generator = ImageGeneration(prompts = prompts, version = VERSION)
    images = image_generator()
    save(f"images_eval_{VERSION}.npy", images)