from PIL import Image
import os

# Obtener el directorio actual donde se encuentra el script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Configurar rutas relativas
carpeta_origen = os.path.join(directorio_actual, 'dataset', 'empty')
carpeta_destino = os.path.join(directorio_actual, 'dataset', 'sand')

# Crear carpeta destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Coordenadas originales del JSON
x = float(252.03)
y = float(604.90)
width = float(321.20)
height = float(538.37)

# Ajustar posición (valores negativos mueven izquierda/arriba)
move_left = 170  # Píxeles a mover a la izquierda
move_up = 280    # Píxeles a mover arriba

# Aplicar ajustes
x = max(0, x - move_left)  # Mover a la izquierda
y = max(0, y - move_up)    # Mover arriba

# Procesar imágenes
for archivo in os.listdir(carpeta_origen):
    if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
        ruta_imagen = os.path.join(carpeta_origen, archivo)
        imagen = Image.open(ruta_imagen)

        # Obtener dimensiones de la imagen
        ancho_imagen, alto_imagen = imagen.size

        # Calcular límites derecho e inferior
        right = min(x + width, ancho_imagen)
        lower = min(y + height, alto_imagen)

        # Asegurar que las coordenadas estén dentro de los límites de la imagen
        x_crop = max(0, x)
        y_crop = max(0, y)

        # Crear el recorte
        try:
            imagen_recortada = imagen.crop((
                int(round(x_crop)),
                int(round(y_crop)),
                int(round(right)),
                int(round(lower))
            ))

            # Guardar imagen recortada
            ruta_imagen_recortada = os.path.join(carpeta_destino, archivo)
            imagen_recortada.save(ruta_imagen_recortada)
            print(f"Recortada y guardada exitosamente: {ruta_imagen_recortada}")
            print(f"Coordenadas de recorte usadas: x={x_crop}, y={y_crop}, right={right}, lower={lower}")
        
        except Exception as e:
            print(f"Error procesando {archivo}: {str(e)}")