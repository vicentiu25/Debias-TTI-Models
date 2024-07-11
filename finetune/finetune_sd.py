from test_text_to_image import TextToImage, TextToImageSDXL
import unittest
import logging
import sys
import logging.config
import json

FORMATTER = logging.Formatter('"%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"')

logger = logging.getLogger(__name__) 
sys.stdout.flush()
console_handler = logging.StreamHandler()
console_handler.setFormatter(FORMATTER)
logger.addHandler(console_handler) 
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    unittest.main(argv=[''], defaultTest='TextToImage.test_text_to_image')