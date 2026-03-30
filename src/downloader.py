import os
import urllib.request
import tarfile
import zipfile

MODELS_DIR = "models"

# URLs de los modelos
# EfficientDet-Lite
TFLITE_MODEL_URL = "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite"

# SSD MobileNetV2 320x320
TF_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

# Archivos de destino
TFLITE_MODEL_FILE = os.path.join(MODELS_DIR, "efficientdet_lite0.tflite")
TF_MODEL_TAR = os.path.join(MODELS_DIR, "ssd_mobilenet_v2.tar.gz")
TF_MODEL_DIR = os.path.join(MODELS_DIR, "ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")


def download_file(url, target_path):
    print(f"Descargando {url} ...")
    urllib.request.urlretrieve(url, target_path)
    print(f"Descargado: {target_path}")

def extract_tar(tar_path, extract_path):
    print(f"Extrayendo {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extracción completa.")

def setup_models():
    """Descarga y extrae los modelos si no existen."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Preparar TFLite
    if not os.path.exists(TFLITE_MODEL_FILE):
        print("Modelo TFLite no encontrado.")
        download_file(TFLITE_MODEL_URL, TFLITE_MODEL_FILE)
    else:
        print("Modelo TFLite ya existe.")

    # Preparar TF
    if not os.path.exists(TF_MODEL_DIR):
        if not os.path.exists(TF_MODEL_TAR):
            print("Modelo TF no encontrado.")
            download_file(TF_MODEL_URL, TF_MODEL_TAR)
        extract_tar(TF_MODEL_TAR, MODELS_DIR)
        # Limpiar
        os.remove(TF_MODEL_TAR)
    else:
        print("Modelo TF ya existe.")

if __name__ == "__main__":
    setup_models()
