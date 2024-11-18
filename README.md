# YOLO Swimming Pose Estimation

This repository contains code for training and validating a YOLO pose estimation model (yolo11n-pose.pt) specifically for swimming analysis.

## Project Structure
```
.
├── data/
│   ├── video/                     # Swimming videos (.webm format)
│   │   ├── Side_above_water/
│   │   ├── Side_underwater/
│   │   └── Side_water_level/
│   │       └── [Swimmer_Parameters]/  # Various swimmer configurations
│   ├── labels/                    # Corresponding 2D joint labels
│   │   ├── Aerial/
│   │   ├── Front/
│   │   ├── Side_above_water/
│   │   ├── Side_underwater/
│   │   └── Side_water_level/
│   │       └── [Swimmer_Parameters]/  # Matching label configurations
│   └── test.mov                   # Test video for prediction
├── models/                        # Trained models directory
├── predictions/                   # Output directory for predictions
├── train.py                      # Training script
├── predict.py                    # Prediction script
└── requirements.txt              # Python dependencies
```

## Dataset Structure
The dataset is organized hierarchically:
1. Camera Angles:
   - Aerial
   - Front
   - Side_above_water
   - Side_underwater
   - Side_water_level

2. Swimmer Parameters:
   - Skin (0.25, 0.75)
   - Muscle (2, 8)

3. Water Parameters (where applicable):
   - Quantity (0.25, 0.75)
   - Height (0.6, 1.0, 1.5)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yolo-pose-swimXYZ.git
cd yolo-pose-swimXYZ
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare your dataset:
   - Ensure videos (.webm) are in the correct subdirectories under `data/video/`
   - Ensure corresponding labels are in matching subdirectories under `data/labels/`
   - Place test video as `data/test.mov`

## Training

Run the training script:
```bash
python train.py
```

The script automatically:
- Discovers all videos and their corresponding labels
- Splits the dataset into train/val sets (80/20)
- Creates necessary configuration files
- Trains with minimal hyperparameters for quick testing:
  - 1 epoch
  - Batch size: 4
  - Image size: 640x640
  - Learning rate: 0.001

Trained models are saved in:
- `models/` directory (exported models)
- `runs/pose/` directory (training artifacts)

## Prediction

To run prediction on a test video:
```bash
python predict.py
```

This will:
1. Load the latest trained model
2. Process data/test.mov
3. Save predictions to predictions/test_predictions.avi
4. Save detection labels to predictions/labels/
5. Print evaluation metrics including:
   - Average detections per frame
   - Inference speed
   - Processing statistics

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics
- OpenCV
- 16GB RAM minimum

## Notes
- The training script automatically handles the nested directory structure
- Videos and labels must maintain the same relative path structure
- For quick testing, hyperparameters are set to minimal values
- Increase epochs and adjust other parameters for better performance
