from PIL import Image
import os

# Configure folder paths
carpeta_origen = '/Users/juanpablocabreraquiroga/Downloads/03 Dataset/03 Empty'
carpeta_destino = '/Users/juanpablocabreraquiroga/Downloads/03 Dataset/04 Sand'

if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Original coordinates from JSON
x = float(252.03)
y = float(604.90)
width = float(321.20)
height = float(538.37)

# Adjust position (negative values move left/up)
move_left = 170  # Pixels to move left
move_up = 280    # Pixels to move up

# Apply adjustments
x = max(0, x - move_left)  # Move left
y = max(0, y - move_up)    # Move up

# Process images
for archivo in os.listdir(carpeta_origen):
    if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
        ruta_imagen = os.path.join(carpeta_origen, archivo)
        imagen = Image.open(ruta_imagen)

        # Get image dimensions
        ancho_imagen, alto_imagen = imagen.size

        # Calculate right and lower boundaries
        right = min(x + width, ancho_imagen)
        lower = min(y + height, alto_imagen)

        # Ensure coordinates are within image bounds
        x_crop = max(0, x)
        y_crop = max(0, y)

        # Create the crop
        try:
            imagen_recortada = imagen.crop((
                int(round(x_crop)),
                int(round(y_crop)),
                int(round(right)),
                int(round(lower))
            ))

            # Save cropped image
            ruta_imagen_recortada = os.path.join(carpeta_destino, archivo)
            imagen_recortada.save(ruta_imagen_recortada)
            print(f"Successfully cropped and saved: {ruta_imagen_recortada}")
            print(f"Crop coordinates used: x={x_crop}, y={y_crop}, right={right}, lower={lower}")
        
        except Exception as e:
            print(f"Error processing {archivo}: {str(e)}")