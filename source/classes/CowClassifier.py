import torch
import torch.nn as nn

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
    