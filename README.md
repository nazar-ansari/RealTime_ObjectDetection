
![Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ64Qu9KhTyG6XEeTzDGiYHoofGmTGJNQk53A77VG-LyQ&s)


# _RealTime Object Detection_

Object Detection System: A repository for real-time detection and classification of objects using state-of-the-art machine learning techniques. This system can accurately detect and classify objects in images and videos, making it useful for a wide range of applications such as surveillance, robotics, and autonomous driving.

## Documentation

[API Reference Documentation for YOLOv8](https://docs.ultralytics.com/hub/inference_api/)


## Usage/Examples

```bash
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')
```

