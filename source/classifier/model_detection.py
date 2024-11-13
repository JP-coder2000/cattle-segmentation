import os
import torch
from torchvision import transforms
from PIL import Image
from source.classifier.model_architecture import CowClassifier

os.listdir()


# Set device
device = torch.device('mps' if torch.mps.is_available() else 'cpu')


# Load the full model
model = torch.load("models/cow_class_model_.pth")
model.to(device)
model.eval()

# Transform for test images (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# # Class labels
class_labels = {0: 'vaca_acostada', 1: 'vaca_de_pie'}

# # Directory to process
test_directory = "test_img/vaca_acostada"  # Set to the directory containing images for prediction

# # Counters for prediction results
total_images = 0
class_counts = {label: 0 for label in class_labels.keys()}

# # Process each image in the directory
for filename in os.listdir(test_directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        img_path = os.path.join(test_directory, filename)
        
        try:
            # Open and transform the image
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            # Run the model on the image
            output = model(image)
            _, pred_label = output.max(1)
            pred_label = pred_label.item()
            
            # Update counts and display result
            class_counts[pred_label] += 1
            total_images += 1
            print(f"Image: {filename} - Predicted class: {class_labels[pred_label]}")      
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# # Summary
print("\nPrediction Summary:")
print(f"Total images processed: {total_images}")
for label, count in class_counts.items():
    print(f"Number of images in '{class_labels[label]}' class: {count}")
