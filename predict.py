import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import glob

def get_latest_model():
    """Get the path to the latest trained model"""
    models_dir = Path('models')
    if not models_dir.exists():
        # Check runs directory if models directory is empty
        runs_dir = Path('runs/pose')
        if not runs_dir.exists():
            raise FileNotFoundError("No trained models found. Please run training first.")
        
        latest_run = max(runs_dir.glob('train*'), key=os.path.getctime)
        weights_file = latest_run / 'weights' / 'best.pt'
    else:
        # Get latest model from models directory
        model_files = list(models_dir.glob('*.pt'))
        if not model_files:
            raise FileNotFoundError("No model files found in models directory")
        weights_file = max(model_files, key=os.path.getctime)
    
    if not weights_file.exists():
        raise FileNotFoundError(f"No weights file found in {weights_file.parent}")
    
    return str(weights_file)

def predict_video():
    # Load the latest trained model
    try:
        model_path = get_latest_model()
        print(f"Using model: {model_path}")
        model = YOLO(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Input video path
    video_path = 'data/test.mov'
    if not os.path.exists(video_path):
        print(f"Error: Test video not found at {video_path}")
        return

    # Create output directory if it doesn't exist
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True)
    
    # Output video path
    output_path = output_dir / 'test_predictions.avi'

    # Run prediction with tracking
    results = model.predict(
        source=video_path,
        save=True,
        save_txt=True,
        stream=True,
        conf=0.5,
        line_width=2,
        show=False,
        device='cpu',  # Use 'cuda' if GPU available
        project=str(output_dir),
        name='predictions'
    )

    # Print prediction metrics
    print("\nPrediction Results:")
    total_frames = 0
    total_detections = 0
    total_time = 0

    for r in results:
        total_frames += 1
        if r.boxes is not None:
            total_detections += len(r.boxes)
        if hasattr(r, 'speed'):
            total_time += r.speed['inference']

    if total_frames > 0:
        print(f"Processed frames: {total_frames}")
        print(f"Average detections per frame: {total_detections/total_frames:.2f}")
        print(f"Average inference time: {total_time/total_frames:.1f}ms per frame")

    print(f"\nPrediction video saved to: {output_path}")
    print(f"Prediction labels saved to: {output_dir}/labels/")

if __name__ == '__main__':
    predict_video()
