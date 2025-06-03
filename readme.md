--- README.md ---
# Custom Object Detection (Electric Box & Human)

This project trains a custom object detection model using TensorFlow and OpenCV. No pre-trained models are used.

## Structure
- `utils/`: Utility modules for data loading, augmentation, modeling, and visualization
- `main.py`: Main training and evaluation script

## Setup
```bash
pip install tensorflow opencv-python albumentations scikit-learn matplotlib
```

## Usage
1. Place your frames in `./frames` and annotation XML files in `./annotations`
2. Run training:
```bash
python main.py
```
3. Visual results will be shown after training.

## Requirements
- Python 3.8+
- TensorFlow 2.8+

--- utils/__init__.py ---
# empty file for utils package
