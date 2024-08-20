import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


class NumpyArrayDataset(Dataset):
    def __init__(self, array_dict):
        """
        Args:
            arrays (list of np.ndarray): List of NumPy arrays.
            keys (list): List of keys corresponding to each array.
        """
        self.arrays = array_dict
        #self.keys = keys

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: Contains the 'array' and 'key' for the item at index idx.
        """
        array = self.arrays[idx][1]
        key = self.arrays[idx][0]
        # Define the torchvision image transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        input_tensor = transform(array)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        image_float_np = array.astype(np.float32) / 255.0
        # Add a batch dimension
        #input_tensor = input_tensor.unsqueeze(0)

        return {'tensor': input_tensor, 'array':image_float_np, 'key': key}

