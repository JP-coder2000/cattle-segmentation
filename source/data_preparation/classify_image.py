import os
import cv2
import shutil

# Obtener el directorio actual donde se encuentra el script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Carpetas de origen y destino (relativas al directorio del script)
source_folder = os.path.join(directorio_actual, "fotos_vacas")
output_folders = {
    "vaca_de_pie": os.path.join(directorio_actual, "fotos_vacas", "vaca_de_pie"),
    "vaca_acostada": os.path.join(directorio_actual, "fotos_vacas", "vaca_acostada"),
    "cama_vacia": os.path.join(directorio_actual, "fotos_vacas", "cama_vacia"),
}

# Crear carpetas de destino si no existen
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Obtener todas las imágenes de la carpeta de origen
image_files = [
    f for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Para rastrear las imágenes movidas y sus ubicaciones anteriores
history = []

# Función para mostrar imágenes con instrucciones y moverlas a la carpeta correspondiente
def classify_images():
    for image_file in image_files:
        image_path = os.path.join(source_folder, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error al cargar la imagen {image_file}")
            continue

        # Agregar la guía de teclas a la imagen
        instructions = (
            "1: Vaca de pie, 2: Vaca acostada, 3: Cama vacia, Esc: Salir, z: Cancelar"
        )
        cv2.putText(
            img,
            instructions,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Mostrar la imagen con las instrucciones
        cv2.imshow("Clasifica la imagen", img)
        key = cv2.waitKey(0)  # Esperar a que el usuario presione una tecla

        # Opciones de clasificación
        if key == ord("1"):
            destination_folder = output_folders["vaca_de_pie"]
        elif key == ord("2"):
            destination_folder = output_folders["vaca_acostada"]
        elif key == ord("3"):
            destination_folder = output_folders["cama_vacia"]
        elif key == 27:  # Código ASCII para la tecla Esc
            print("Saliendo del programa...")
            break
        elif key == ord("z"):  # Código para cancelar la última clasificación
            if history:
                last_image, last_dest = history.pop()
                shutil.move(last_dest, os.path.join(source_folder, last_image))
                print(
                    f"Cancelada la clasificación de {last_image}. Movida de vuelta a {source_folder}."
                )
            else:
                print("No hay imágenes para cancelar.")
            cv2.destroyAllWindows()
            continue
        else:
            print(f"Clasificación inválida para {image_file}. Imagen omitida.")
            continue

        # Mover la imagen a la carpeta correspondiente y guardar en el historial
        destination_path = os.path.join(destination_folder, image_file)
        shutil.move(image_path, destination_path)
        history.append((image_file, destination_path))
        print(f"Moviendo {image_file} a {destination_folder}")

        # Cerrar la ventana de la imagen
        cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_images()