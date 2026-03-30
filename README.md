# TensorFlow_vs_TensorFlow-Lite ****En UBUNTU****
Una aplicación de detección de objetos en tiempo real diseñada para comparar el rendimiento entre modelos regulares de **TensorFlow** y modelos optimizados con **TensorFlow Lite**.   El proyecto proporciona una interfaz visual (HUD) con métricas de rendimiento en vivo como FPS, tiempo de inferencia y consumo de memoria. 
=======
# Comparacion , TensorExpo Object Detection 

Una aplicación de detección de objetos en tiempo real diseñada para comparar el rendimiento entre modelos regulares de **TensorFlow** y modelos optimizados con **TensorFlow Lite**. 

El proyecto proporciona una interfaz visual (HUD) con métricas de rendimiento en vivo como FPS, tiempo de inferencia y consumo de memoria. Esto resulta ideal para evaluar implementaciones de IA en entornos con diferentes capacidades de cómputo (por ejemplo, Edge vs Desktop).

## Características Principales

- **Detección en Tiempo Real:** Detección de objetos sobre el flujo de una o varias cámaras web.
- **Soporte Multi-Backend:** Ejecuta inferencias usando TensorFlow (modelo estándar) o TensorFlow Lite (modelo optimizado para Edge).
- **Modo Paralelo (Comparativa):** Compara ambos motores de inferencia (TF vs TFLite) lado a lado sobre el mismo flujo de video para analizar de forma directa las diferencias de latencia y consumo de recursos.
- **Inferencia en Imágenes Estáticas:** Permite procesar y probar el modelo sobre imágenes (con sistema de autodescarga de imágenes de prueba).
- **HUD de Rendimiento:** Superposición en pantalla de métricas en vivo (Uso de Memoria, FPS globales, latencia de inferencia).
- **Menú Interactivo (`run.sh`):** Automatiza la creación del entorno virtual, la instalación de dependencias, la descarga de los modelos pre-entrenados y la selección de la cámara.

## Requisitos Previos

- Python 3.8+
- Utilidad `python3-venv` instalada en tu sistema.
- Conexión a internet (para descargar los modelos y las dependencias la primera vez).

## Instalación y Ejecución

El proyecto está diseñado para funcionar directamente desde su menú interactivo:

1. Clona el repositorio e ingresa a la carpeta:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd TensorExpo
   ```

2. Ejecuta el script principal:
   ```bash
   ./run.sh
   ```

El script `run.sh` se encarga de todo el proceso de arranque:
- Crea un entorno virtual (`venv/`).
- Instala todas las librerías necesarias especificadas en `requirements.txt`.
- Descarga automáticamente los modelos `.pb` (TensorFlow) y `.tflite` (TensorFlow Lite) a la carpeta `models/\`.
- Proporciona un menú en la terminal para elegir qué modo ejecutar y qué cámara utilizar.

## Estructura del Proyecto

- `main.py`: Punto de entrada de la aplicación en Python. Gestiona los argumentos, los bucles de video y el enrutamiento.
- `run.sh`: Menú Bash interactivo y gestor de arranque.
- `src/`: Lógica principal agrupada:
  - `detector_tf.py` y `detector_tflite.py`: Clases encargadas de encapsular la inferencia de ambos frameworks.
  - `downloader.py`: Utilidad para adquirir los modelos pre-entrenados de internet si no existen.
  - `camera_utils.py`: Auxiliar para listar cámaras conectadas.
  - `utils.py`: Funciones para dibujar cajas de detección (bounding boxes) y la interfaz gráfica HUD.
- `requirements.txt`: Dependencias de Python (TensorFlow, OpenCV, psutil, etc.).
