import os
import cv2
import glob
import shutil
from pathlib import Path

def extract_frames(video_path, output_dir):
    """Extract frames from video and save as jpg"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame as jpg
        frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    return frame_paths

def process_videos():
    """Process all videos and create frame-label mapping"""
    # Get current working directory
    cwd = os.getcwd()
    
    # Create frames directory
    frames_dir = os.path.join(cwd, 'data', 'frames')
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)  # Clear previous frames
    os.makedirs(frames_dir)
    
    # Get all .webm files recursively
    video_files = glob.glob('data/video/**/*.webm', recursive=True)
    print(f"\nFound {len(video_files)} .webm files")
    
    # Dictionary to store frame-label mappings
    mappings = []
    
    for video_path in video_files:
        # Convert video path to corresponding label path
        label_path = video_path.replace('video/', 'labels/')
        label_path = label_path.replace('.webm', '/COCO/2D_cam.txt')
        
        if os.path.exists(label_path):
            # Extract frames from video
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(frames_dir, video_name)
            print(f"\nExtracting frames from {video_path}")
            
            extracted_frames = extract_frames(video_path, video_frames_dir)
            
            if extracted_frames:
                # Add frame-label mapping
                abs_label_path = os.path.join(cwd, label_path)
                for frame_path in extracted_frames:
                    mappings.append({
                        'frame': frame_path,
                        'label': abs_label_path
                    })
                print(f"Extracted {len(extracted_frames)} frames")
            else:
                print(f"Warning: No frames extracted from {video_path}")
        else:
            print(f"\nWarning: No matching label found for video: {video_path}")
            print(f"Expected label path: {label_path}")
    
    # Save mappings to file
    mappings_file = os.path.join(cwd, 'frame_label_mappings.txt')
    with open(mappings_file, 'w') as f:
        for mapping in mappings:
            f.write(f"{mapping['frame']},{mapping['label']}\n")
    
    print(f"\nProcessing complete:")
    print(f"- Processed {len(video_files)} videos")
    print(f"- Extracted {len(mappings)} total frames")
    print(f"- Frame-label mappings saved to: {mappings_file}")
    print(f"- Frames saved in: {frames_dir}")

if __name__ == '__main__':
    process_videos()
