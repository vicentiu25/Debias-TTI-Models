from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Optional, Any, Dict
from PIL import Image
import scipy.io
import numpy as np
import os

class MpiiDataset(Dataset):
    """Class for structuring the dataset."""

    def __init__(self,
                 annotations: list[str]):
        """Format dataset.
        
        Args:
            annotations: list of the annotations of dataset.
        """
        self.image = self.get_images(annotations)
        self.text = self.get_texts(annotations)
    
    def __len__(self) -> int:
        """
        Compute the length of the dataset (number of samples).

        :return: The length of the dataset.
        """
        return len(self.text)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        return {'image' : self.image[index],
                'text' : self.text[index]}
    
    def get_images(self, annotations: list[str]) -> list[int]:
        release_ann = annotations['RELEASE']
        image_list = []
        for ann in release_ann['annolist'][0][0]['image'][0]:
            image_filename = ann['name'][0, 0][0]
            image_path = '../datasets/mpii_dataset/images/' + image_filename
            image_list.append(image_path)
            
        return image_list
    
    def get_texts(self, annotations: list[str]) -> list[str]:
        release_ann = annotations['RELEASE']
        text_list = []

        for ann in release_ann['act'][0,0]:
            category_label = ann[0][1]
            text_list.append(category_label)

        return text_list
        
class MpiiDataModule:
    """Module for loading dataset"""
    
    def __init__(self):
        """Loads dataset"""
        annotations = scipy.io.loadmat('../datasets/mpii_dataset/mpii_human_pose_v1_u12_1.mat')
        self.dataset: Optional[Dataset] = MpiiDataset(annotations)
