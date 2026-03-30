import sys
import cv2
import time
import psutil
import os

from src.downloader import setup_models, TFLITE_MODEL_FILE, TF_MODEL_DIR
from src.detector_tf import TFDetector
from src.detector_tflite import TFLiteDetector
from src.utils import draw_hud, draw_boxes

def process_frame(frame, detector, backend_name, process, prev_time):
    """
    Coordina la inferencia, cálculo de rendimiento y dibujo de resultados sobre un frame.
    
    Args:
        frame: Imagen capturada (OpenCV BGR).
        detector: Instancia de TFDetector o TFLiteDetector.
        backend_name: Nombre del motor ("TF" o "TFLITE").
        process: Objeto psutil para medir recursos del sistema.
        prev_time: Marca de tiempo del frame anterior para calcular FPS.
        
    Returns:
        frame_procesado: Frame con HUD y cajas de detección.
        curr_time: Marca de tiempo actual para el siguiente ciclo.
    """
    # Aquí se llama al método detect() que contiene la lógica de inferencia
    detections, inf_time = detector.detect(frame)
    
    # Calcular FPS globales de la aplicación (no solo de la inferencia)
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    
    # Dibujar las cajas de detección y etiquetas sobre el frame
    frame = draw_boxes(frame, detections, backend_name)
    
    # Superponer el HUD (Heads-Up Display) con estadísticas de rendimiento
    frame = draw_hud(frame, fps, backend_name, process, inf_time, detector.memory_footprint_mb)
    
    return frame, curr_time


def run_realtime(backend, camera_index=0):

    print(f"Iniciando modo tiempo real con backend: {backend} (Cámara ID: {camera_index})")
    process = psutil.Process() # Para monitorear RAM/CPU
    
    # Inicializar el detector correspondiente
    detector = TFDetector(TF_MODEL_DIR) if backend == 'tf' else TFLiteDetector(TFLITE_MODEL_FILE)
    
    # Abrir la captura de la webcam (índice proporcionado)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error al abrir la cámara web.")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar el frame actual
        frame, prev_time = process_frame(frame, detector, backend.upper(), process, prev_time)
        
        # Mostrar el resultado visual si no estamos en un entorno sin GUI
        if not os.environ.get('NO_GUI'):
            cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Object Detection', frame)
            
            # Gestionar salida del bucle (tecla 'q' o cerrar ventana)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def run_parallel(camera_index=0):
    """
    Modo comparativo: Ejecuta simultáneamente TF y TFLite sobre el mismo flujo de video.
    Ideal para comparar latencia y precisión lado a lado.
    """
    print(f"Iniciando modo paralelo (TF vs TFLite) (Cámara ID: {camera_index})...")
    process = psutil.Process()
    # Cargamos ambos motores en memoria
    detector_tf = TFDetector(TF_MODEL_DIR)
    detector_tflite = TFLiteDetector(TFLITE_MODEL_FILE)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error al abrir la cámara web.")
        return

    prev_time_tf = time.time()
    prev_time_tflite = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Duplicamos el frame para procesar cada uno con un motor distinto
        frame_tf = frame.copy()
        frame_tflite = frame.copy()

        # Inferencia TF (Estandard)
        frame_tf, prev_time_tf = process_frame(frame_tf, detector_tf, "TF", process, prev_time_tf)
        
        # Inferencia TFLite (Optimizado/Edge)
        frame_tflite, prev_time_tflite = process_frame(frame_tflite, detector_tflite, "TFLITE", process, prev_time_tflite)

        # Concatenar ambos resultados horizontalmente
        combined_frame = cv2.hconcat([frame_tflite, frame_tf])
        
        if not os.environ.get('NO_GUI'):
            cv2.namedWindow('TF vs TFLite Comparison', cv2.WINDOW_NORMAL)
            cv2.imshow('TF vs TFLite Comparison', combined_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            if cv2.getWindowProperty('TF vs TFLite Comparison', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

def run_image(backend, image_path):
    """
    Procesa una única imagen estática en lugar de video.
    """
    if not image_path or not os.path.exists(image_path):
        print("Ruta de imagen no válida.")
        return

    print(f"Iniciando modo imagen con backend {backend}")
    process = psutil.Process()
    detector = TFDetector(TF_MODEL_DIR) if backend == 'tf' else TFLiteDetector(TFLITE_MODEL_FILE)
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error al cargar la imagen.")
        return

    # Inferencia única
    prev_time = time.time()
    frame, _ = process_frame(frame, detector, backend.upper(), process, prev_time)
    
    if not os.environ.get('NO_GUI'):
        cv2.namedWindow('Image Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Image Detection', frame)
        print("Cierra la ventana gráfica o presiona cualquier tecla en ella para salir.")
        
        # Mantener la ventana abierta hasta interacción del usuario
        while cv2.getWindowProperty('Image Detection', cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
                
        cv2.destroyAllWindows()
    else:
        print("Inferencia completada en modo NO_GUI.")

def main():
    """
    Punto de entrada principal. Gestiona la configuración de modelos y el enrutamiento de modos.
    """
    # 1. Asegurar que los modelos están descargados
    setup_models()

    # 2. Validar argumentos mínimos
    if len(sys.argv) < 2:
        print("Uso: python main.py <modo> [backend] [ruta_imagen]")
        return

    # 3. Leer modo y backend de los argumentos de línea de comandos
    mode = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "tflite"

    # 4. Enrutamiento según el modo seleccionado
    if mode == "realtime":
        camera_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        run_realtime(backend, camera_index)
    elif mode == "parallel":
        camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        run_parallel(camera_index)
    elif mode == "image":
        # Para el modo imagen se requiere un tercer argumento: la ruta del archivo
        image_path = sys.argv[3] if len(sys.argv) > 3 else None
        run_image(backend, image_path)
    else:
        print(f"Modo '{mode}' no reconocido.")

if __name__ == "__main__":
    main()
