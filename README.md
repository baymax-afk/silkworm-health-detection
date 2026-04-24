# Silkworm Health Detection

A YOLO-based deep learning model designed to detect and classify infected and healthy silkworms in real-time.

## Project Overview

This project uses YOLOv11 (You Only Look Once) for real-time object detection to identify diseased and healthy silkworms. The model is trained on a custom dataset and can perform inference on images, videos, and live webcam feeds. This tool is useful for silkworm farmers and researchers to monitor silkworm health and detect diseases early.

## Features

- **Real-time Detection**: Detect infected and healthy silkworms in live webcam feeds
- **Video Processing**: Analyze silkworm health in video recordings
- **Image Inference**: Run predictions on single images with confidence scores
- **High Accuracy**: YOLOv11 backbone for fast and accurate detection
- **Easy to Use**: Simple API powered by Ultralytics

## Model Files

- `silk_disease_det.pt` - Pre-trained model for detecting infected and healthy silkworms
- `yolo11n.pt` - YOLOv11 nano base model (for training from scratch)

## Installation

```bash
pip install ultralytics opencv-python
```

## Training the Model

If you want to train your own model from scratch or fine-tune the existing one:

```python
from ultralytics import YOLO

# Load a pretrained model (recommended)
model = YOLO("yolo11n.pt")  # options: yolo8n.pt, yolo11s.pt, yolo11m.pt, etc.

# Train the model
model.train(
    data="data.yaml",      # path to your dataset config file
    epochs=50,             # number of training epochs
    imgsz=640,             # image size
    batch=16,              # batch size
    device=0               # 0 = GPU, 'cpu' for CPU training
)

# Save/export model (optional)
model.export(format="onnx")  # export to ONNX format
```

## Usage

### 1. Predict on Single Image

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("silk_disease_det.pt")

# Run prediction on an image
results = model.predict(
    source="test.jpg",     # path to image
    conf=0.5,              # confidence threshold
    save=True              # saves output image
)

# Print results
for r in results:
    print(r.boxes)
```

### 2. Real-Time Webcam Detection

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("silk_disease_det.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model(frame)

    # Loop through detections
    for r in results:
        for box in r.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class + confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Show frame
    cv2.imshow("Silkworm Disease Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. Predict on Video

```python
from ultralytics import YOLO

model = YOLO("silk_disease_det.pt")

# Run inference on video file
model.predict(
    source="video.mp4",    # path to your video
    conf=0.4,              # confidence threshold
    save=True,             # save output video
    show=True              # display live results
)
```

## Dataset Structure

Your `data.yaml` should follow this format:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ['healthy', 'infected']  # class names
```

## Performance

The model achieves high accuracy in detecting:
- **Healthy Silkworms**: Normal, disease-free specimens
- **Infected Silkworms**: Diseased specimens with visible symptoms

## Requirements

- Python 3.8+
- ultralytics
- opencv-python
- torch
- torchvision

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs and feature requests.