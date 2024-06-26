import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_imgs = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        image_path = self.all_imgs[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image



def load_data(dataset_dir, batch_size, shuffle=True):
    # Define the transformation
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        # transforms.Normalize((0.7427, 0.6436, 0.6116), (0.2755, 0.2848, 0.2512))
    ])

    # Create the dataset
    train_data_tensor = CustomDataSet(dataset_dir, transform=transform)

    # Create a DataLoader
    return DataLoader(train_data_tensor, batch_size=batch_size, shuffle=shuffle, drop_last=True)