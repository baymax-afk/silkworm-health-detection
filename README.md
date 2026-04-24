# Silkworm Health Detection

A YOLO-based deep learning model designed to detect and classify infected and healthy silkworms in real-time.

## Project Overview

This project uses YOLO26n (You Only Look Once) for real-time object detection to identify diseased and healthy silkworms. The model is trained on a custom dataset and can perform inference on images, videos, and live webcam feeds. This tool is useful for silkworm farmers and researchers to monitor silkworm health and detect diseases early.

## Features

- **Real-time Detection**: Detect infected and healthy silkworms in live webcam feeds
- **Video Processing**: Analyze silkworm health in video recordings
- **Image Inference**: Run predictions on single images with confidence scores
- **High Accuracy**: YOLOv11 backbone for fast and accurate detection
- **Easy to Use**: Simple API powered by Ultralytics

## Model Files

- `silk_disease_det.pt` - Pre-trained model for detecting infected and healthy silkworms
- `yolo11n.pt` - YOLOv11 nano base model (for training from scratch)


[The dataset](https://universe.roboflow.com/capstone-sucsf/silkworm-dataset/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

## Installation

```bash
pip install ultralytics opencv-python
```

## Usage
Example usages have been provided in the the main.ipynb file

* Predict on Single Image
* Real-Time Webcam Detection
* Predict on Video



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
