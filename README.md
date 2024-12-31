# Emotion Detection System

This project implements a Convolutional Neural Network (CNN) for detecting human emotions from facial expressions using the FER-2013 dataset.

## Features
- Real-time emotion detection from webcam feed
- Support for image-based emotion detection
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python train_model.py
```

3. For real-time detection:
```bash
python real_time_detection.py
```

## Project Structure
- `train_model.py`: Script for training the CNN model
- `real_time_detection.py`: Script for real-time emotion detection
- `model.py`: CNN model architecture
- `utils.py`: Utility functions for data processing
