"""
Convert MAT dataset images to video
Converts img001001.jpg to img002300.jpg into a single video file
"""

import cv2
import os
from pathlib import Path
import re


# ==========================================================
# CONFIGURATION
# ==========================================================
IMAGES_DIR = "D:/stgnn_project/data/mat_dataset/images"
OUTPUT_DIR = "D:/stgnn_project/data/videos"
OUTPUT_FILENAME = "mat_dataset_full.mp4"

# Video settings
FPS = 10  # Frames per second (adjust as needed: 5=slow, 10=normal, 30=fast)
START_FRAME = 1001  # img001001.jpg
END_FRAME = 2300    # img002300.jpg


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def get_frame_number(filename):
    """Extract frame number from filename like img001234.jpg"""
    match = re.search(r'img(\d{6})\.jpg', filename)
    if match:
        return int(match.group(1))
    return None


def generate_frame_list(start, end):
    """Generate list of frame filenames in order."""
    frames = []
    for i in range(start, end + 1):
        filename = f"img{i:06d}.jpg"
        frames.append(filename)
    return frames


# ==========================================================
# VIDEO CREATION
# ==========================================================
def create_video():
    """Create video from image sequence."""
    print("=" * 70)
    print("MAT DATASET TO VIDEO CONVERTER")
    print("=" * 70)
    
    # Validate input directory
    if not os.path.exists(IMAGES_DIR):
        print(f"[ERROR] Images directory not found: {IMAGES_DIR}")
        return
    
    print(f"[OK] Images directory: {IMAGES_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output directory: {OUTPUT_DIR}\n")
    
    # Generate frame list
    print(f"[INFO] Generating frame list ({START_FRAME} to {END_FRAME})...")
    frame_files = generate_frame_list(START_FRAME, END_FRAME)
    total_frames = len(frame_files)
    print(f"[OK] Total frames: {total_frames}\n")
    
    # Check if first frame exists
    first_frame_path = os.path.join(IMAGES_DIR, frame_files[0])
    if not os.path.exists(first_frame_path):
        print(f"[ERROR] First frame not found: {first_frame_path}")
        print(f"Please verify:")
        print(f"  1. Images are in: {IMAGES_DIR}")
        print(f"  2. Format is: img001001.jpg, img001002.jpg, etc.")
        return
    
    # Read first frame to get dimensions
    print("[INFO] Reading first frame to get dimensions...")
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"[ERROR] Cannot read first frame: {first_frame_path}")
        return
    
    height, width = first_frame.shape[:2]
    print(f"[OK] Video dimensions: {width}x{height}")
    print(f"[OK] Frame rate: {FPS} FPS\n")
    
    # Calculate video duration
    duration_seconds = total_frames / FPS
    duration_mins = duration_seconds / 60
    print(f"[INFO] Expected video duration: {duration_mins:.2f} minutes ({duration_seconds:.1f} seconds)\n")
    
    # Initialize video writer
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"[INFO] Creating video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    if not out.isOpened():
        print("[ERROR] Failed to initialize video writer")
        return
    
    # Process frames
    print("\n[PROCESSING FRAMES]")
    print("-" * 70)
    
    frames_written = 0
    frames_skipped = 0
    
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(IMAGES_DIR, frame_file)
        
        # Check if file exists
        if not os.path.exists(frame_path):
            frames_skipped += 1
            if frames_skipped <= 5:  # Only show first 5 warnings
                print(f"  [WARNING] Frame not found: {frame_file}")
            continue
        
        # Read frame
        frame = cv2.imread(frame_path)
        
        if frame is None:
            frames_skipped += 1
            if frames_skipped <= 5:
                print(f"  [WARNING] Cannot read: {frame_file}")
            continue
        
        # Write to video
        out.write(frame)
        frames_written += 1
        
        # Progress indicator (every 100 frames)
        if (idx + 1) % 100 == 0 or idx == len(frame_files) - 1:
            progress = (idx + 1) / total_frames * 100
            print(f"  Progress: {idx+1}/{total_frames} ({progress:.1f}%) | "
                  f"Written: {frames_written} | Skipped: {frames_skipped}", end='\r')
    
    # Release video writer
    out.release()
    
    print("\n" + "-" * 70)
    
    # Final statistics
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Video created successfully!")
        print("=" * 70)
        print(f"Output file: {output_path}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Frames written: {frames_written}/{total_frames}")
        print(f"Frames skipped: {frames_skipped}")
        print(f"Resolution: {width}x{height}")
        print(f"Frame rate: {FPS} FPS")
        print(f"Duration: {frames_written/FPS:.1f} seconds ({frames_written/FPS/60:.2f} minutes)")
        print("=" * 70)
        
        # Suggestions
        print("\n[NEXT STEPS]")
        print(f"1. Play video: Open {output_path} in VLC or any video player")
        print(f"2. Use in pipeline: Update run_pipeline_realtime.py:")
        print(f'   VIDEO_SOURCE = r"{output_path}"')
        print("=" * 70)
    else:
        print("\n[ERROR] Video file was not created")


def verify_images():
    """Quick verification of image availability."""
    print("\n[VERIFICATION MODE]")
    print("Checking first 10 and last 10 frames...\n")
    
    # Check first 10
    print("First 10 frames:")
    for i in range(START_FRAME, START_FRAME + 10):
        filename = f"img{i:06d}.jpg"
        path = os.path.join(IMAGES_DIR, filename)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {filename}")
    
    # Check last 10
    print("\nLast 10 frames:")
    for i in range(END_FRAME - 9, END_FRAME + 1):
        filename = f"img{i:06d}.jpg"
        path = os.path.join(IMAGES_DIR, filename)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {filename}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    """Main execution."""
    import sys
    
    # Check for verification mode
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_images()
        return
    
    try:
        create_video()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Video creation stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
