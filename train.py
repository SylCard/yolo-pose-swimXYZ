import os
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

def load_frame_label_mappings():
    """Load frame-label mappings from file"""
    mappings_file = 'frame_label_mappings.txt'
    if not os.path.exists(mappings_file):
        raise FileNotFoundError(
            "frame_label_mappings.txt not found! "
            "Please run preprocess_videos.py first to extract frames."
        )
    
    frames = []
    labels = []
    with open(mappings_file, 'r') as f:
        for line in f:
            frame_path, label_path = line.strip().split(',')
            if os.path.exists(frame_path):
                frames.append(frame_path)
                labels.append(label_path)
            else:
                print(f"Warning: Frame not found: {frame_path}")
    
    return frames, labels

def setup_dataset_yaml():
    """Create YAML file for dataset configuration"""
    frame_paths, label_paths = load_frame_label_mappings()
    
    if not frame_paths:
        raise ValueError(
            "No valid frame-label pairs found! "
            "Please run preprocess_videos.py to extract frames."
        )
    
    # Split into train/val (80/20)
    split_idx = int(len(frame_paths) * 0.8)
    train_frames = frame_paths[:split_idx]
    val_frames = frame_paths[split_idx:]
    
    # Get absolute path for current directory
    cwd = os.getcwd()
    train_list_path = os.path.join(cwd, 'train_list.txt')
    val_list_path = os.path.join(cwd, 'val_list.txt')
    
    # Write train/val lists
    with open(train_list_path, 'w') as f:
        f.write('\n'.join(train_frames))
    with open(val_list_path, 'w') as f:
        f.write('\n'.join(val_frames))
    
    yaml_content = f"""
path: {cwd}  # root dir (absolute path)
train: {train_list_path}  # train frames list (absolute path)
val: {val_list_path}  # val frames list (absolute path)

# Keypoints
kpt_shape: [17, 3]  # number of keypoints, number of dims (x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes
names:
  0: swimmer

# Dataset structure info
nc: 1  # number of classes
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset configuration:")
    print(f"Found {len(train_frames)} training frames and {len(val_frames)} validation frames")
    print(f"First training frame: {train_frames[0] if train_frames else 'None'}")
    print(f"First validation frame: {val_frames[0] if val_frames else 'None'}")
    print(f"Dataset root: {cwd}")
    print(f"Train list: {train_list_path}")
    print(f"Val list: {val_list_path}")

def train():
    # Load the YOLO model
    model = YOLO('yolo11n-pose.pt')

    # Setup dataset configuration
    setup_dataset_yaml()

    # Training arguments with minimal settings for quick testing
    args = {
        'data': 'dataset.yaml',
        'epochs': 1,
        'imgsz': 640,
        'batch': 4,
        'device': 'cpu',  # Use 'cuda' if GPU available
        'workers': 1,
        'patience': 50,
        'lr0': 0.001,
        'name': f'swim_pose_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'cache': True,  # Cache images for faster training
        'verbose': True,
        'task': 'pose'  # Explicitly set task as pose estimation
    }

    # Train the model
    results = model.train(**args)

    # Print training results
    print("\nTraining Results:")
    print(f"mAP50-95: {results.maps[0]:.3f}")
    print(f"Precision: {results.fitness:.3f}")
    print(f"Recall: {results.results_dict['metrics/recall50-95(B)']:.3f}")

    # Save the trained model
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / f"swim_pose_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model.export(format='pt', save_dir=str(model_path))
    print(f"\nModel saved to: {model_path}")

if __name__ == '__main__':
    train()
