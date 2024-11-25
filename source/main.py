import os
import sys
import torch
import pathlib
import platform
import numpy as np
from PIL import Image
from model_utils import setup_logger

# Inicializar el logger
logger = setup_logger()

# Detectar sistema operativo
os_name = platform.system()

if os_name == "Windows":
    # Configuración para Windows
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath 
    logger.info("Sistema operativo: Windows - Configuración de pathlib.Path a WindowsPath")
elif os_name in ("Linux", "Darwin"):  # Darwin es para macOS
    # Configuración para sistemas tipo Unix (Linux, macOS)
    temp = pathlib.PosixPath
    pathlib.WindowsPath = pathlib.PosixPath
    logger.info(f"Sistema operativo: {os_name} - Configuración de pathlib.Path a PosixPath")
else:
    log_msg = f"Sistema operativo no soportado: {os_name}"
    logger.error(log_msg)
    raise OSError(log_msg)

# Obtener el directorio del archivo actual
file_dir = os.getcwd()

# Subir al directorio raíz del proyecto
project_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(project_dir)

# Importar módulos y funciones
from classifier.model_architecture import CowClassifier
from database.db_utils import add_image_info, add_cow_detail

try:
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar de detección
    model_detection = torch.hub.load('ultralytics/yolov5', 'custom', '../models/bounding/best.pt')
    logger.info("Modelo de detección cargado correctamente.")

    # Cargar modelo de clasificación
    model_classification = CowClassifier().to(device)

    # Cargar los pesos en el modelo de clasificación
    model_classification.load_state_dict(torch.load('../models/classifier/cow_class_model_state.pth', map_location=device))
    model_classification.eval()
    logger.info("Modelo de clasificación cargado correctamente.")
except Exception as e:
    logger.error(f"Error al cargar los modelos: {e}")
    raise e

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

try:
    # Inicialización de la base de datos
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    logger.info("Conexión a base de datos incializada.")
except Exception as e:
    logger.error(f"Error al conectar con la base de datos: {e}")
    raise e

def detect_objects(directory):
    """
    Performs object detection and posture classification on the specified images
    """
    logger.info(f"Iniciando procesamiento de detección y clasificación en imágenes del directorio {directory}")
    predictions = {}
    posture_counts = {'vaca_acostada': 0, 'vaca_de_pie': 0, 'clasificación_fallida': 0}
    
    for filename in os.listdir(directory):   
        try:
            if filename.endswith(('.png', '.jpg', '.jpeg')):

                existing_image = session.query(ImageInfo).filter_by(file_name=filename).first()
                if existing_image:
                    logger.info(f"Ya existe una predicción para el archivo {filename}. Omitiendo predicción...")
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
                    
                log_msg = (f"Imagen procesada: {filename} | Vacas detectadas: {len(detections)}")
                logger.info(log_msg)
                
                predictions[filename] = detections

        except Exception as e:
            logger.error(f"Error al procesar el archivo {filename}: {str(e)}")
    
    logger.info("Procesamiento de imágenes completado.")
    return predictions

DIRECTORY = '../dataset/bounding/detect/'

# Ejecutar la detección y obtener los resultados
results = detect_objects(DIRECTORY)