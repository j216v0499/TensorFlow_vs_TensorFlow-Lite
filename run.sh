#!/bin/bash

# Script de entorno para TensorExpo Object Detection

VENV_DIR="venv"

# Comprobar si existe el entorno virtual
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual en $VENV_DIR..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Error: No se pudo crear el entorno virtual. Asegúrate de tener python3-venv instalado."
        exit 1
    fi
    echo "Entorno virtual creado exitosamente."
fi

# Activar el entorno virtual
source $VENV_DIR/bin/activate

# Instalar dependencias
if ! python -c "import cv2, tensorflow, psutil" &> /dev/null; then
    echo "Instalando/actualizando dependencias..."
    pip install --upgrade pip > /dev/null
    pip install -r requirements.txt > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: No se pudieron instalar las dependencias."
        deactivate
        exit 1
    fi
    echo "Dependencias instaladas exitosamente."
fi

while true; do
    clear
    echo "========================================================"
    echo "    TFLite vs TensorFlow Object Detection Comparison    "
    echo "========================================================"
    echo "1) Tiempo Real - TFLite (Recomendado/Rápido)"
    echo "2) Tiempo Real - TensorFlow (Pesado)"
    echo "3) Comparativa Paralela (TFLite vs TensorFlow)"
    echo "4) Procesar imagen de prueba estática (test.jpg)"
    echo "5) Salir"
    echo "========================================================"
    read -p "Elige una opción (1-5): " opcion

    # Preguntar por ID de cámara si es un modo de tiempo real
    if [ "$opcion" == "1" ] || [ "$opcion" == "2" ] || [ "$opcion" == "3" ]; then
        echo "Buscando cámaras disponibles..."
        python3 src/camera_utils.py
        echo ""
        read -p "Introduce el ID de la cámara de la lista anterior [0]: " CAMERA_ID
        CAMERA_ID=${CAMERA_ID:-0}
    fi

    case $opcion in
        1)
            echo "Iniciando TFLite..."
            python main.py realtime tflite "$CAMERA_ID"
            ;;
        2)
            echo "Iniciando TensorFlow..."
            python main.py realtime tf "$CAMERA_ID"
            ;;
        3)
            echo "Iniciando Comparativa..."
            python main.py parallel "$CAMERA_ID"
            ;;
        4)
            if [ ! -d "test_images" ]; then
                echo "Descargando imágenes de prueba..."
                mkdir -p test_images
                wget -q -O test_images/test1.jpg https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg
                wget -q -O test_images/test2.jpg https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg
                wget -q -O test_images/test3.jpg https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image3.jpg
            fi
            
            # Elegir una imagen aleatoria
            IMAGEN_RANDOM=$(ls test_images/*.jpg | sort -R | head -1)
            
            echo "Procesando imagen aleatoria ($IMAGEN_RANDOM) con TFLite y TF..."
            python main.py image tflite "$IMAGEN_RANDOM"
            ;;
        5)
            echo "Saliendo..."
            break
            ;;
        *)
            echo "Opción no válida. Por favor, elige un número del 1 al 5."
            ;;
    esac
    
    if [ "$opcion" != "5" ]; then
        echo ""
        read -p "Presiona Enter para volver al menú principal..."
    fi
done

deactivate
