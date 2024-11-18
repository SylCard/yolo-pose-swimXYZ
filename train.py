import os
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import glob

def get_video_label_pairs():
    """Get matching video and label paths"""
    video_paths = []
    label_paths = []
    
    # Get all .webm files recursively
    video_files = glob.glob('data/video/**/*.webm', recursive=True)
    
    for video_path in video_files:
        # Convert video path to corresponding label path
        label_path = video_path.replace('video/', 'labels/').replace('.webm', '.txt')
        if os.path.exists(label_path):
            video_paths.append(video_path)
            label_paths.append(label_path)
    
    return video_paths, label_paths

def setup_dataset_yaml():
    """Create YAML file for dataset configuration"""
    video_paths, label_paths = get_video_label_pairs()
    
    # Split into train/val (80/20)
    split_idx = int(len(video_paths) * 0.8)
    train_videos = video_paths[:split_idx]
    val_videos = video_paths[split_idx:]
    
    # Write train/val lists
    with open('train_list.txt', 'w') as f:
        f.write('\n'.join(train_videos))
    with open('val_list.txt', 'w') as f:
        f.write('\n'.join(val_videos))
    
    yaml_content = f"""
path: {os.getcwd()}  # dataset root dir
train: train_list.txt  # train videos list
val: val_list.txt  # val videos list

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
    
    print(f"Found {len(train_videos)} training videos and {len(val_videos)} validation videos")

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
        'verbose': True
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
