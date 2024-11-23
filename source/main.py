import os
import sys
import torch
import pathlib
import platform
import numpy as np
from PIL import Image
from pathlib import Path

# Detectar sistema operativo
os_name = platform.system()

if os_name == "Windows":
    # Configuración para Windows
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath 
    print("Sistema operativo: Windows - Configuración de pathlib.Path a WindowsPath")
elif os_name in ("Linux", "Darwin"):  # Darwin es para macOS
    # Configuración para sistemas tipo Unix (Linux, macOS)
    temp = pathlib.PosixPath
    pathlib.WindowsPath = pathlib.PosixPath
    print(f"Sistema operativo: {os_name} - Configuración de pathlib.Path a PosixPath")
else:
    raise OSError(f"Sistema operativo no soportado: {os_name}")

# Obtener el directorio del notebook
notebook_dir = os.getcwd()

# Subir al directorio raíz del proyecto
project_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(project_dir)

# Importar módulos y funciones
from classifier.model_architecture import CowClassifier
from database.db_utils import add_image_info, add_cow_detail

# Configuración de directorios
SAVE_DIR = '../results/classifier/cut_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar de detección
model_detection = torch.hub.load('ultralytics/yolov5', 'custom', '../models/bounding/best.pt')

# Cargar modelo de clasificación
model_classification = CowClassifier().to(device)

# Cargar los pesos en el modelo de clasificación
model_classification.load_state_dict(torch.load('../models/classifier/cow_class_model_state.pth', map_location=device))
model_classification.eval()

from datetime import datetime
from model_utils import calculate_centroid, crop_and_save_detection
from database.db_utils import add_image_info, add_cow_detail
from database.db_init import ImageInfo
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar URL de la base de datos
DB_URL = os.getenv('DB_FULL_URL')

# Inicialización de la base de datos
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
session = Session()

def detect_objects(directory):
    """
    Performs object detection and posture classification on the specified images
    """
    predictions = {}
    posture_counts = {'vaca_acostada': 0, 'vaca_de_pie': 0, 'clasificación_fallida': 0}
    
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):

            existing_image = session.query(ImageInfo).filter_by(file_name=filename).first()
            if existing_image:
                print(f"Ya existe una predicción para el archivo {filename}. Omitiendo predicción...")
                continue  # Si ya existe, omite el procesamiento

            img_path = os.path.join(directory, filename)
            original_img = np.array(Image.open(img_path))
            
            # Redimensionar para la detección
            img_resized = Image.open(img_path).resize((640, 320))
            img_resized_array = np.array(img_resized)

            # Realizar la detección
            results = model_detection(img_resized_array)
            results_df = results.pandas().xyxy[0]
            
            # Factores de escala
            scale_x = original_img.shape[1] / 640
            scale_y = original_img.shape[0] / 320
            
            detections = []
            
            # Obtener el timestamp actual (fecha + hora)
            processed_at = datetime.now()
            cow_count = len(results_df)

            # Añadir la información de la imagen a la base de datos
            image_id = add_image_info(processed_at, cow_count, filename)
            
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
                
                # Guardar detalles de la vaca en la base de datos
                add_cow_detail(
                    id_img_fk=image_id,
                    centroid=centroid,
                    accuracy=detection['confidence'],
                    posture=posture
                )
                
                # Guardar información de la detección en una lista
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
                
                print(
                    f"Cow {idx + 1}:\n"
                    f"  Bounding Box Coordinates:\n"
                    f"    - xmin: {detection_info['xmin']:.2f}\n"
                    f"    - ymin: {detection_info['ymin']:.2f}\n"
                    f"    - xmax: {detection_info['xmax']:.2f}\n"
                    f"    - ymax: {detection_info['ymax']:.2f}\n"
                    f"  Centroid:\n"
                    f"    - x: {detection_info['centroid_x']:.2f}\n"
                    f"    - y: {detection_info['centroid_y']:.2f}\n"
                    f"  Confidence: {detection_info['confidence']:.2f}\n"
                    f"  Posture: {posture}\n"
                )
            
            predictions[filename] = detections
    
    return predictions

DIRECTORY = '../dataset/bounding/detect/'

# Ejecutar la detección y obtener los resultados
results = detect_objects(DIRECTORY)