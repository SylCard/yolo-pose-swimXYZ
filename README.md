# YOLO Swimming Pose Estimation

This repository contains code for training and validating a YOLO pose estimation model (yolo11n-pose.pt) specifically for swimming analysis. Due to the large dataset and frame extraction process, training is recommended on a machine with sufficient computational power.

## Hardware Requirements

Recommended specifications:
- High-performance CPU for frame extraction
- GPU with 8GB+ VRAM for training
- 32GB+ RAM
- 500GB+ free disk space for frames and models

Minimum specifications:
- 16GB RAM
- 100GB free disk space
- Note: Training on CPU-only machines is possible but will be very slow

## Project Structure
```
.
├── data/
│   ├── video/                                 # Swimming videos (.webm format)
│   │   └── [Nested structure of videos]
│   ├── labels/                                # COCO format keypoint labels
│   │   └── [Matching structure with 2D_cam.txt files]
│   └── test.mov                               # Test video for prediction
├── models/                                    # Trained models directory
├── predictions/                               # Output directory for predictions
├── preprocess_videos.py                       # Video frame extraction script
├── train.py                                  # Training script
├── predict.py                                # Prediction script
└── requirements.txt                          # Python dependencies
```

## Quick Start

1. Clone and setup:
```bash
git clone https://github.com/yourusername/yolo-pose-swimXYZ.git
cd yolo-pose-swimXYZ
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
# For CUDA-enabled machines (recommended):
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For macOS:
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
pip install -r requirements.txt
```

3. Prepare dataset:
   - Place .webm videos in data/video/ following the directory structure
   - Place corresponding labels in data/labels/
   - Place test video as data/test.mov

4. Run the pipeline:
```bash
# Step 1: Extract frames (takes significant time and disk space)
python preprocess_videos.py

# Step 2: Train the model (GPU recommended)
python train.py

# Step 3: Run prediction
python predict.py
```

## Dataset Organization

The dataset follows a hierarchical structure:
- Camera Angles: Side_above_water, Side_underwater, Side_water_level
- Swimmer Parameters: Skin (0.25, 0.75), Muscle (2, 8)
- Water Parameters: Quantity (0.25, 0.75), Height (0.6, 1.0, 1.5)
- Lighting: rotx (110, 140), roty (190, 280, 360)
- Speed: Speed_2, Speed_3
- Positions: position_1,75, position_3,75

## Pipeline Details

1. Preprocessing (preprocess_videos.py):
   - Extracts frames from all .webm videos
   - Creates frame-label mappings file
   - WARNING: Requires significant disk space
   - Run this once, preferably on a machine with fast storage

2. Training (train.py):
   - Uses preprocessed frames
   - Current settings optimized for quick testing:
     * 1 epoch
     * Batch size: 4
     * Image size: 640x640
   - Modify these parameters for full training

3. Prediction (predict.py):
   - Runs inference on test.mov
   - Saves predictions as AVI file
   - Outputs detection metrics

## Performance Notes

- Frame extraction is I/O intensive - use SSD if possible
- Training benefits significantly from GPU acceleration
- Default parameters are minimal for testing
- For full training, consider adjusting:
  * epochs (50-100)
  * batch size (8-16 with GPU)
  * image size (up to 1280)
  * learning rate (0.001 is default)

## Files in .gitignore

The following are not tracked in git:
- data/ (dataset and extracted frames)
- models/ (trained models)
- predictions/ (output files)
- frame_label_mappings.txt
- dataset.yaml
- train/val lists
- Python/environment files

## Troubleshooting

1. Out of memory during preprocessing:
   - Delete data/frames/ if it exists
   - Try processing fewer videos initially
   - Use a machine with more RAM

2. Training too slow:
   - Ensure GPU is being utilized if available
   - Reduce batch size if memory is an issue
   - Consider using a more powerful machine

3. Disk space issues:
   - Ensure at least 100GB free space
   - Clean data/frames/ between runs if needed
   - Consider processing subsets of videos

For full training on the complete dataset, a CUDA-enabled GPU and significant RAM is highly recommended.
