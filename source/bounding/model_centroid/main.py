import torch
import os
import numpy as np
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', '../../results/bounding/weights/best.pt')  # Load YOLOv5s

def detect_objects(directory):
    """
    Performs object detection on the specified images using the loaded YOLOv5 model.
    Args:
        directory: The path to the directory containing images to process.
    Returns:
        predictions: Dictionary containing detection results for each image, including coordinates.
    """
    predictions = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).resize((640, 320))
            img = np.array(img)

            # Realizar la detección
            results = model(img)
            
            # Extraer las coordenadas y otros datos de detección
            detections = []
            
            # Convertir los resultados a un formato pandas para fácil manipulación
            results_df = results.pandas().xyxy[0]
            
            # Iterar sobre cada detección
            for idx, detection in results_df.iterrows():
                detection_info = {
                    'xmin': detection['xmin'],
                    'ymin': detection['ymin'],
                    'xmax': detection['xmax'],
                    'ymax': detection['ymax'],
                    'confidence': detection['confidence'],
                    'class': detection['class'],
                    'name': detection['name']
                }
                detections.append(detection_info)
            
            # Guardar las detecciones para esta imagen
            predictions[filename] = detections
            
            # Mostrar la imagen con las detecciones
            results.show()
            
            # Imprimir las coordenadas detectadas
            print(f"\nDetecciones para {filename}:")
            for det in detections:
                print(f"Bounding Box: xmin={det['xmin']:.2f}, ymin={det['ymin']:.2f}, "
                      f"xmax={det['xmax']:.2f}, ymax={det['ymax']:.2f}, "
                      f"confidence={det['confidence']:.2f}, class={det['name']}")
        
    return predictions

DIRECTORY = '../../dataset/bounding/detect/'

# Ejecutar la detección y obtener los resultados
results = detect_objects(DIRECTORY)