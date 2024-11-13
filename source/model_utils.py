import os
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from classifier.model_architecture import CowClassifier

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar modelo de clasificación
model = CowClassifier().to(device)

# Cargar los pesos en el modelo
model.load_state_dict(torch.load('../models/classifier/cow_class_model_state.pth', map_location=device))
model.eval()

# Transformación para el modelo de clasificación
transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Etiquetas de clase
class_labels = {0: 'vaca_acostada', 1: 'vaca_de_pie'}

# Configuración de directorios
SAVE_DIR = '../results/classifier/cut_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
def classify_cow_posture(img_pil):
    """
    Classifies whether a cow is standing or lying down
    Args:
        img_pil: PIL Image of the cow
    Returns:
        predicted class label
    """
    try:
        # Preparar la imagen para la clasificación
        img_tensor = transform_classification(img_pil).unsqueeze(0).to(device)
        
        # Realizar la predicción
        with torch.no_grad():
            output = model(img_tensor)
            _, pred_label = output.max(1)
            pred_label = pred_label.item()
            
        return class_labels[pred_label]
    except Exception as e:
        print(f"Error en la clasificación: {e}")
        return "clasificación_fallida"
    
def crop_and_save_detection(img, detection, filename, detection_index):
    """
    Crops the image according to the bounding box coordinates, classifies the cow's posture, and saves it
    """
    # Convertir coordenadas a enteros
    xmin = int(detection['xmin'])
    ymin = int(detection['ymin'])
    xmax = int(detection['xmax'])
    ymax = int(detection['ymax'])
    
    # Asegurarse de que las coordenadas están dentro de los límites
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.shape[1], xmax)
    ymax = min(img.shape[0], ymax)
    
    # Recortar la imagen
    cropped_img = img[ymin:ymax, xmin:xmax]
    
    # Convertir a PIL Image
    cropped_img_pil = Image.fromarray(cropped_img)
    
    # Clasificar la postura de la vaca
    posture = classify_cow_posture(cropped_img_pil)
    
    # Generar nombre del archivo incluyendo la postura
    base_name = os.path.splitext(filename)[0]
    save_name = f"{base_name}_detection_{detection_index}_{posture}.jpg"
    save_path = os.path.join(SAVE_DIR, save_name)
    
    # Guardar la imagen
    cropped_img_pil.save(save_path)
    
    return posture

def calculate_centroid(xmin, ymin, xmax, ymax):
    """
    Calculates the centroid of a given bounding box's coordinates
    Args:
        xmin: The x position of the bottom left point
        ymin: The y position of the bottom left point
        xmax: The x position of the top right point
        ymax: The y position of the top right point
    Returns:
        The centroid of the given coordinates
    """
    return [(xmax + xmin) / 2, (ymax + ymin) / 2]