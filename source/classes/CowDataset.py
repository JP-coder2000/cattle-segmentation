from torch.utils.data import Dataset
import os
from PIL import Image

# Custom Dataset class
class CowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Define class labels
        self.class_labels = {'vaca_acostada': 0, 'vaca_de_pie': 1}
        
        # File extensions to consider
        valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        
        # Load images from each numbered folder (1-8)
        for folder_num in range(1, 9):
            folder_path = os.path.join(root_dir, str(folder_num))
            for class_name, label in self.class_labels.items():
                class_folder = os.path.join(folder_path, class_name)
                
                if not os.path.exists(class_folder):
                    continue
                
                for filename in os.listdir(class_folder):
                    if any(filename.lower().endswith(ext) for ext in valid_image_extensions):
                        img_path = os.path.join(class_folder, filename)
                        try:
                            img = Image.open(img_path).convert("RGB")
                            self.data.append((img_path, label))
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label