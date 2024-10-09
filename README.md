# Análisis del Comportamiento de Vacas en Espacios de Descanso Utilizando Imágenes Aéreas

## Descripción General del Proyecto

Este proyecto tiene como objetivo analizar el comportamiento de vacas lecheras en sus espacios de descanso a través de imágenes aéreas. Utilizamos minería de datos para encontrar patrones y hallazgos significativos respecto al uso de las camas de arena, como el tiempo que pasan descansando, la frecuencia con la que prefieren ciertos espacios, y las posturas adoptadas en las camas.

## Objetivos del Proyecto

### Objetivos de Negocio
- **Eficiencia del espacio de descanso**: Mejorar el uso de las camas disponibles para optimizar el descanso de las vacas.
- **Facilitar la toma de decisiones**: Proporcionar insights que permitan al equipo tomar decisiones más informadas sobre la gestión del espacio de descanso.
  
### Preguntas clave:
1. ¿Cuántas vacas están descansando en las camas?
2. ¿Cuánto tiempo permanecen en las camas?
3. ¿Qué camas son más utilizadas que otras?
4. ¿Cómo se distribuyen las posiciones de las vacas al descansar (paradas o acostadas)?

### Objetivos de Minería de Datos
- Clasificar imágenes aéreas para identificar si una cama contiene una vaca parada, una vaca acostada o está vacía.
- Obtener métricas como el tiempo promedio de uso de cama por vaca, el porcentaje de uso de cada cama y el estado de las vacas.

## Metodología

Este proyecto sigue la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining), abarcando las siguientes fases:

1. **Entendimiento del Negocio**: Definir los objetivos de negocio y las preguntas clave.
2. **Entendimiento de los Datos**: Recopilar y analizar las imágenes para comprender su estructura y contenido.
3. **Preparación de los Datos**: Clasificar manualmente las imágenes, etiquetando cada cama según la presencia y postura de las vacas.
4. **Modelado**: Construir un modelo de clasificación de imágenes utilizando redes neuronales convolucionales (CNN) para detectar si una cama está ocupada o vacía y si la vaca está parada o acostada.
5. **Evaluación**: Validar el rendimiento del modelo y ajustar los parámetros si es necesario.
6. **Despliegue**: Generar reportes y conclusiones sobre los patrones de uso de las camas por las vacas.

## Estructura del Repositorio

```bash
|-- dataset/
|   |-- images/                 # Contiene las imágenes aéreas (1920x1080 px) clasificadas.
|-- notebooks/
|   |-- data_preparation.ipynb   # Código para la preparación y exploración de los datos.
|   |-- model_training.ipynb     # Notebook con el proceso de entrenamiento del modelo.
|   |-- analysis.ipynb           # Análisis de resultados y hallazgos.
|-- models/
|   |-- cnn_model.h5             # Modelo de clasificación entrenado (CNN).
|-- results/
|   |-- reports/                 # Reportes con los resultados del análisis.
|-- documentation/
|   |-- business/                # Documentación pertinente de la fase de entendimiento de negocio.
|   |-- data/                    # Documentación pertinente de la fase de entendimiento y preparación de datos.
|   |-- modeling/                # Documentación pertinente de la fase de modelado.
|   |-- evaluation/              # Documentación pertinente de la fase de evaluación.
|   |-- deployment/              # Documentación pertinente de la fase de despliegue.
|-- README.md                    # Descripción del proyecto (este archivo).
