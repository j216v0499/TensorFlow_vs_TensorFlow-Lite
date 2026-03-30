import cv2
import psutil
import time
import os

# Cargar labels para ambos formatos
LABELS_80 = [] # Para TFLite (0-indexed indices)
LABELS_90 = [] # Para TensorFlow (1-indexed IDs)

def load_labels():
    global LABELS_80, LABELS_90
    try:
        # Lista de 80 clases reales (0-79)
        with open(os.path.join(os.path.dirname(__file__), "coco_labels.txt"), "r") as f:
            LABELS_80 = [line.strip() for line in f.readlines()]
    except:
        print("Warning: coco_labels.txt no encontrado.")

    try:
        # Lista de 90 clases (con gaps n/a)
        with open(os.path.join(os.path.dirname(__file__), "efficientdet_lite0_labels.txt"), "r") as f:
            LABELS_90 = [line.strip() for line in f.readlines()]
    except:
        print("Warning: efficientdet_lite0_labels.txt no encontrado.")

load_labels()

def get_class_name(class_id, backend_name="TFLITE"):
    try:
        idx = int(class_id)
        # Tanto TFLite (EfficientDet-Lite0) como TensorFlow (SSD MobileNet) 
        # en este proyecto usan el mapa de 91 clases (con gaps) de COCO.
        if 0 <= idx < len(LABELS_90):
            name = LABELS_90[idx]
            if name and name != "n/a":
                return name
    except Exception:
        pass
    return str(class_id)


def draw_hud(frame, fps, backend_name, process, inference_time_ms, memory_info_mb):
    """
    Dibuja un HUD con información de rendimiento y fondo translúcido.
    """
    height, width = frame.shape[:2]
    
    # Obtener uso de CPU
    cpu_usage = psutil.cpu_percent()

    # Configuración de estilo
    x, y = 10, 30
    line_spacing = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    color = (0, 255, 0) # Verde para el texto
    bg_color = (0, 0, 0) # Negro para el fondo
    alpha = 0.6 # Factor de transparencia (0.0 totalmente transparente, 1.0 opaco)

    texts = [
        f"Backend: {backend_name}",
        f"FPS: {fps:.1f}",
        f"Latencia inf: {inference_time_ms:.1f} ms",
        f"CPU: {cpu_usage}%",
        f"Model RAM: {memory_info_mb:.1f} MB"
    ]

    # 1. Determinar el área total del HUD para el fondo
    max_w = 0
    total_h = len(texts) * line_spacing
    for text in texts:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        if tw > max_w: max_w = tw

    # 2. Crear un overlay para la transparencia
    overlay = frame.copy()
    padding = 10
    cv2.rectangle(overlay, (x - padding, y - 25), 
                  (x + max_w + padding, y + total_h - 10), bg_color, cv2.FILLED)
    
    # 3. Mezclar el overlay con el frame original
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 4. Dibujar el texto sobre el fondo translúcido
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (x, y + i * line_spacing), 
                    font, font_scale, color, font_thickness, cv2.LINE_AA)
        
    return frame

def draw_boxes(frame, detections, backend_name="TFLite", threshold=0.5):
    """
    Dibuja los bounding boxes en el frame.
    Detections es una lista de diccionarios: {'ymin', 'xmin', 'ymax', 'xmax', 'score', 'class'}
    """
    height, width = frame.shape[:2]
    
    for det in detections:
        if det['score'] >= threshold:
            ymin, xmin, ymax, xmax = det['ymin'], det['xmin'], det['ymax'], det['xmax']
            
            # Convertir coordenadas relativas a absolutas
            (left, right, top, bottom) = (int(xmin * width), int(xmax * width), 
                                          int(ymin * height), int(ymax * height))
            
            # Dibujar caja
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Etiqueta amigable
            class_name = get_class_name(det['class'], backend_name)
            label = f"{class_name} ({int(det['score'] * 100)}%)"
            
            # Fondo para el texto
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (left, top - th - 10), (left + tw, top), (0, 0, 0), cv2.FILLED)
            
            cv2.putText(frame, label, (left, top - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return frame
