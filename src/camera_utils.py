
import os

def get_camera_list():
    cameras = []
    base_path = "/sys/class/video4linux"
    if not os.path.exists(base_path):
        return cameras
    
    devices = sorted([d for d in os.listdir(base_path) if d.startswith("video")])
    for dev in devices:
        name_path = os.path.join(base_path, dev, "name")
        if os.path.exists(name_path):
            with open(name_path, "r") as f:
                name = f.read().strip()
            # En Linux, video0 y video1 suelen ser el mismo dispositivo físico (formatos distintos)
            # Filtramos para mostrar solo los que tienen capacidades de captura reales si es posible,
            # pero por ahora mostramos todos los videoX únicos por nombre o simplemente todos.
            index = int(dev.replace("video", ""))
            cameras.append((index, name))
    
    # Eliminar duplicados de nombre seguidos (típico de video0/video1)
    unique_cameras = []
    seen_names = set()
    for idx, name in cameras:
        if name not in seen_names:
            unique_cameras.append((idx, name))
            seen_names.add(name)
            
    return unique_cameras

if __name__ == "__main__":
    cam_list = get_camera_list()
    if not cam_list:
        print("No se detectaron cámaras.")
    else:
        print("Cámaras detectadas:")
        for idx, name in cam_list:
            print(f"{idx}: {name}")
