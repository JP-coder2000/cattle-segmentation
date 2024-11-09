import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import random


# Define the model
class CowClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CowClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: [32, 256, 256]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: [64, 128, 128]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: [128, 64, 64]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # Output: [256, 32, 32]
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))  # Output: [256,1,1]

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        # print(f"After pool1: {x.shape}")
        x = self.pool2(torch.relu(self.conv2(x)))
        # print(f"After pool2: {x.shape}")
        x = self.pool3(torch.relu(self.conv3(x)))
        # print(f"After pool3: {x.shape}")
        x = self.pool4(torch.relu(self.conv4(x)))
        # print(f"After pool4: {x.shape}")
        x = self.global_pool(x)
        # print(f"After global pooling: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"After flattening: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Custom Dataset class
class CowDataset(Dataset):
    def __init__(self, root_dir, images_per_class=None, transform=None):
        self.root_dir = root_dir
        self.images_per_class = images_per_class
        self.transform = transform
        self.data = []
        
        # Define class labels
        self.class_labels = {'vaca_acostada': 0, 'vaca_de_pie': 1}
        
        # File extensions to consider
        valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        
        # Initialize class image lists
        class_image_paths = {label: [] for label in self.class_labels.values()}
        
        # Load images from each numbered folder (1-8)
        for folder_num in range(1, 9):
            folder_path = os.path.join(self.root_dir, str(folder_num))
            for class_name, label in self.class_labels.items():
                class_folder = os.path.join(folder_path, class_name)
                
                if not os.path.exists(class_folder):
                    continue
                
                for filename in os.listdir(class_folder):
                    if any(filename.lower().endswith(ext) for ext in valid_image_extensions):
                        img_path = os.path.join(class_folder, filename)
                        class_image_paths[label].append(img_path)
        
        # Limit the number of images per class
        for label, image_paths in class_image_paths.items():
            if self.images_per_class is not None:
                if len(image_paths) > self.images_per_class:
                    image_paths = random.sample(image_paths, self.images_per_class)
            for img_path in image_paths:
                self.data.append((img_path, label))
        
        # Shuffle the data
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label