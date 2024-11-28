from PIL import Image
import os
import numpy as np
from collections import defaultdict
import shutil

def calculate_brightness(image_path):
    """Calculate the average brightness of an image."""
    with Image.open(image_path) as img:
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array and calculate mean brightness 
        np_img = np.array(img)
        return np.mean(np_img)

def categorize_brightness(brightness):
    """
    Categorize brightness into time of day.
    You can adjust these thresholds based on your specific images.
    """
    if brightness < 85:  # Very dark images
        return "night"
    elif brightness < 140:  # Moderately bright images
        return "morning"
    else:  # Very bright images
        return "afternoon"

def group_images_by_brightness(input_folder, output_base_folder):
    """Group images by their brightness levels and copy them to respective folders."""
    
    # Create output base folder if it doesn't exist
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    # Dictionary to store image paths grouped by brightness category
    brightness_groups = defaultdict(list)
    
    # Dictionary to store actual brightness values for each image
    brightness_values = {}
    
    # Process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            
            # Calculate brightness
            brightness = calculate_brightness(image_path)
            brightness_values[filename] = brightness
            
            # Categorize and group
            category = categorize_brightness(brightness)
            brightness_groups[category].append(filename)
    
    # Create category folders and copy images
    for category, images in brightness_groups.items():
        # Create category folder
        category_folder = os.path.join(output_base_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        
        # Sort images by brightness within each category
        images.sort(key=lambda x: brightness_values[x])
        
        # Copy images to their respective folders
        for img in images:
            src = os.path.join(input_folder, img)
            dst = os.path.join(category_folder, img)
            shutil.copy2(src, dst)
            
        print(f"{category}: {len(images)} images (brightness range: "
              f"{brightness_values[images[0]]:.2f} - {brightness_values[images[-1]]:.2f})")

# Obtener el directorio actual donde se encuentra el script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Configurar rutas relativas
input_folder = os.path.join(directorio_actual, 'dataset', 'sand')
output_base_folder = os.path.join(directorio_actual, 'dataset', 'light')

# Ejecutar el agrupamiento
group_images_by_brightness(input_folder, output_base_folder)