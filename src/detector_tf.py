import tensorflow as tf
import time
import psutil
import os

class TFDetector:
    def __init__(self, model_dir):
        print("Cargando modelo TensorFlow...")
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        self.model = tf.saved_model.load(model_dir)
        self.infer = self.model.signatures["serving_default"]
        
        mem_after = process.memory_info().rss
        self.memory_footprint_mb = max((mem_after - mem_before) / (1024 * 1024), 10.0) # Al menos 10MB
        print(f"Modelo TensorFlow cargado. Memoria aportada: {self.memory_footprint_mb:.1f} MB")
        
    def detect(self, image):
        """
        Realiza la inferencia con el modelo TensorFlow.
        image: Frame BGR de OpenCV.
        Retorna la lista de detecciones procesadas y el tiempo de inferencia en ms.
        """
        # Preparar la imagen (TensorFlow Object Detection API espera un tensor uint8 de tipo [1, None, None, 3])
        # Primero convertir BGR (OpenCV) a RGB
        image_rgb = image[:, :, ::-1] # BGR to RGB
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = input_tensor[tf.newaxis, ...] # Añadir batch dimension

        start_time = time.time()
        
        # === PUNTO DE INFERENCIA (TENSORFLOW) ===
        # Se aplica el modelo cargado sobre el tensor de imagen
        output_dict = self.infer(input_tensor)
        inference_time_ms = (time.time() - start_time) * 1000.0

        num_detections = int(output_dict.pop('num_detections'))
        
        # Procesar salidas
        # Las key de output_dict tienen prefijo en TF2 object detection, 
        # pero dependiendeo del modelo puede ser 'detection_boxes', etc.
        # Asumimos que son tensores float flotantes por defecto de MobileNet v2.
        boxes = output_dict['detection_boxes'][0].numpy()
        classes = output_dict['detection_classes'][0].numpy().astype(int)
        scores = output_dict['detection_scores'][0].numpy()

        detections = []
        for i in range(num_detections):
            if scores[i] > 0.1: # Guardar todo lo que tenga score > 0.1, el utils filtrará
                det = {
                    'ymin': boxes[i][0],
                    'xmin': boxes[i][1],
                    'ymax': boxes[i][2],
                    'xmax': boxes[i][3],
                    'class': classes[i] - 1, # TF Object Detection API is 1-indexed for COCO (0 is background)
                    'score': scores[i]
                }
                detections.append(det)

        return detections, inference_time_ms
