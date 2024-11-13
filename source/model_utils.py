import os
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from classifier.model_architecture import CowClassifier

# Transformación para el modelo de clasificación
transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuración de directorios
SAVE_DIR = '../results/classifier/cut_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Etiquetas de clase
class_labels = {0: 'vaca_acostada', 1: 'vaca_de_pie'}

# Cargar modelo de clasificación
model_classification = CowClassifier().to(device)

# Cargar los pesos en el modelo
model_classification.load_state_dict(torch.load('../models/classifier/cow_class_model_state.pth', map_location=device))
model_classification.eval()
    
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
            output = model_classification(img_tensor)
            _, pred_label = output.max(1)
            pred_label = pred_label.item()
            
        return class_labels[pred_label]
    except Exception as e:
        return "Failed to classify".format(e)
    
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
    print(f"Imagen guardada: {save_path} - Postura: {posture}")
    
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

def detect_objects(directory):
    """
    Performs object detection and posture classification on the specified images
    """
    predictions = {}
    posture_counts = {'vaca_acostada': 0, 'vaca_de_pie': 0, 'clasificación_fallida': 0}
    
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            original_img = np.array(Image.open(img_path))
            
            # Redimensionar para la detección
            img_resized = Image.open(img_path).resize((640, 320))
            img_resized_array = np.array(img_resized)

            # Realizar la detección
            results = model_classification(img_resized_array)
            results_df = results.pandas().xyxy[0]
            
            # Factores de escala
            scale_x = original_img.shape[1] / 640
            scale_y = original_img.shape[0] / 320
            
            detections = []
            print(f"\nProcesando imagen: {filename}")
            
            # Procesar cada detección
            for idx, detection in results_df.iterrows():
                # Ajustar coordenadas a escala original
                xmin = detection['xmin'] * scale_x
                ymin = detection['ymin'] * scale_y
                xmax = detection['xmax'] * scale_x
                ymax = detection['ymax'] * scale_y
                
                centroid = calculate_centroid(xmin, ymin, xmax, ymax)
                
                # Recortar, clasificar y guardar la detección
                posture = crop_and_save_detection(original_img, 
                                                {'xmin': xmin, 'ymin': ymin, 
                                                 'xmax': xmax, 'ymax': ymax}, 
                                                filename, idx)
                
                # Actualizar contadores
                posture_counts[posture] += 1
                
                # Guardar información de la detección
                detection_info = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'confidence': detection['confidence'],
                    'class': detection['class'],
                    'name': detection['name'],
                    'posture': posture
                }
                detections.append(detection_info)
                
                print(f"Detección {idx + 1}: "
                      f"BB: ({detection_info['xmin']:.2f}, {detection_info['ymin']:.2f}, "
                      f"{detection_info['xmax']:.2f}, {detection_info['ymax']:.2f}), "
                      f"Centroide: ({detection_info['centroid_x']:.2f}, {detection_info['centroid_y']:.2f}), "
                      f"Confianza: {detection_info['confidence']:.2f}, "
                      f"Postura: {posture}")
            
            predictions[filename] = detections
            results.show()
    
    # Mostrar resumen final
    print("\nResumen de clasificación de posturas:")
    print(f"Total de vacas acostadas: {posture_counts['vaca_acostada']}")
    print(f"Total de vacas de pie: {posture_counts['vaca_de_pie']}")
    if posture_counts['clasificación_fallida'] > 0:
        print(f"Clasificaciones fallidas: {posture_counts['clasificación_fallida']}")
    
    return predictions