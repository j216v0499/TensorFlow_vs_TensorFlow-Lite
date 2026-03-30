import tensorflow as tf
import numpy as np
import time
import cv2
import psutil
import os

class TFLiteDetector:
    def __init__(self, model_path):
        print("Cargando modelo TFLite...")
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Cargar TFLite model y asignar tensores.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        mem_after = process.memory_info().rss
        # A veces el intérprete base requiere muy poco si mmap, pero los tensores pesan.
        self.memory_footprint_mb = max((mem_after - mem_before) / (1024 * 1024), 2.0) # Al menos 2MB

        # Obtener input y output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Obtener tamaño esperado (ej. 320x320)
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        # Verificar si el modelo está cuantizado o espera floats
        self.is_floating_model = (self.input_details[0]['dtype'] == np.float32)

        print(f"Modelo TFLite cargado. Input shape: {self.width}x{self.height}, Float: {self.is_floating_model}")

    def detect(self, image):
        """
        Realiza la inferencia con el modelo TFLite.
        image: Frame BGR de OpenCV.
        Retorna la lista de detecciones procesadas y el tiempo de inferencia en ms.
        """
        # BGR a RGB y redimensionar
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height))

        # EfficientDet-Lite0 expects uint8 rgb input natively.
        input_data = np.expand_dims(image_resized, axis=0)

        if self.is_floating_model:
            # Normalizar si espera floats (ej. MobileNet float)
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # EfficientDet-Lite0 usually requires uint8
            input_data = np.uint8(input_data)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        start_time = time.time()
        # === PUNTO DE INFERENCIA (TFLITE) ===
        # Se invoca al intérprete para procesar los datos de entrada
        self.interpreter.invoke() # ← aquí ocurre la magia
        inference_time_ms = (time.time() - start_time) * 1000.0

        # EfficientDet-Lite0 outputs:
        # [0]: Bounding boxes (1, 25, 4) - [ymin, xmin, ymax, xmax] formato absoluto o relativo?
        # Normalmente tflite object_detection:
        # TENSOR 0: locations (1, N, 4) or (1, N, 4) normalized
        # TENSOR 1: classes (1, N)
        # TENSOR 2: scores (1, N)
        # TENSOR 3: num_detections (1)

        # Buscamos los tensores de salida (puede variar el orden, los buscaremos por nombre/forma)
        # Asumiendo el orden clásico de TF Lite (boxes, classes, scores, num_detections):
        #5.Leer la prediccion
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] 
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Hay modelos con 4 salidas (num_detections es el 4to)
        if len(self.output_details) > 3:
             num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        else:
             num_detections = len(scores)

        detections = []
        for i in range(num_detections):
             if scores[i] > 0.1: # Filtro base
                 # Las cajas suelen ser [ymin, xmin, ymax, xmax] relativas entre 0 y 1
                 # o a veces multiplicadas por pixel dependiendo del modelo.
                 # EfficientDet-Lite tflite box values are relative 0-1 if they are normalized, 
                 # o son en píxeles si no. Asumiremos relativas y revisamos
                 # EfficientDet-Lite tflite returns classes in 0-index without background label usually
                 det = {
                     'ymin': boxes[i][0],
                     'xmin': boxes[i][1],
                     'ymax': boxes[i][2],
                     'xmax': boxes[i][3],
                     'class': int(classes[i]),
                     'score': scores[i]
                 }
                 # Normalización de boxes si es necesario. (Algunos tflite exportados devuelven valores mayores a 1)
                 if det['ymax'] > 1.0 or det['xmax'] > 1.0:
                      det['ymin'] /= self.height
                      det['ymax'] /= self.height
                      det['xmin'] /= self.width
                      det['xmax'] /= self.width

                 detections.append(det)

        return detections, inference_time_ms
