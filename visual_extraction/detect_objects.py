from ultralytics import YOLO

# Load YOLOv8 model globally (once)
model = YOLO("yolov8n.pt")

def detect_objects(image_path, conf_threshold=0.3):
    results = model(image_path)[0]
    labels = [
        model.names[int(cls)]
        for cls, conf in zip(results.boxes.cls, results.boxes.conf)
        if conf > conf_threshold
    ]
    return list(set(labels))  # Remove duplicates
