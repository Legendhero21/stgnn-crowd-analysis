"""
Test YOLOv8s (VisDrone) ONNX detector on aerial dataset sequences.
Generates detection visualizations and per-sequence statistics.
Production-ready with comprehensive error handling.
"""

import os
import sys
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from yolo_detector import YOLODetector
except ImportError:
    print("[ERROR] Cannot import YOLODetector. Ensure yolo_detector.py is in the same directory.")
    sys.exit(1)


# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET_DIR = "D:/stgnn_project/data/txt_dataset/sequences"
OUTPUT_DIR = "D:/stgnn_project/outputs/yolo_visdrone_test"
ONNX_MODEL = "D:/stgnn_project/src/visdrone-yolov8s.onnx"

# Testing setup
TEST_SEQUENCES = ["00010", "00011", "00012"]  # modify as needed
MAX_FRAMES_PER_SEQ = 10  # Set to None for all frames
DISPLAY_FRAMES = True
SAVE_FRAMES = True
SAVE_DETECTIONS = True

# Detection thresholds (tuned for aerial detection)
CONF_THRESHOLD = 0.10  # Lower for small aerial objects
IOU_THRESHOLD = 0.45


# ==========================================================
# VALIDATION
# ==========================================================
def validate_setup():
    """Validate all required paths and files exist."""
    print("=" * 60)
    print("YOLOv8s (VisDrone) Aerial Detection Test")
    print("=" * 60)
    
    errors = []
    
    # Check ONNX model
    if not os.path.exists(ONNX_MODEL):
        errors.append(f"ONNX model not found: {ONNX_MODEL}")
        print(f"[ERROR] Model not found!")
        print(f"Expected: {ONNX_MODEL}")
        print(f"Run: python export_yolo_onnx.py first")
    else:
        size_mb = os.path.getsize(ONNX_MODEL) / (1024 * 1024)
        print(f"[OK] Model: {ONNX_MODEL} ({size_mb:.2f} MB)")
    
    # Check dataset directory
    if not os.path.exists(DATASET_DIR):
        errors.append(f"Dataset directory not found: {DATASET_DIR}")
        print(f"[ERROR] Dataset not found: {DATASET_DIR}")
    else:
        print(f"[OK] Dataset: {DATASET_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output: {OUTPUT_DIR}")
    
    if errors:
        print("\n[ERRORS DETECTED]")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    print()
    return True


# ==========================================================
# VISUALIZATION HELPERS
# ==========================================================
def draw_detections(frame, detections):
    """
    Draw bounding boxes with class-specific colors.
    
    Args:
        frame: Input frame
        detections: Array of [x1, y1, x2, y2, conf, cls]
    
    Returns:
        Frame with drawn boxes
    """
    vis_frame = frame.copy()
    
    for x1, y1, x2, y2, conf, cls in detections:
        # Color coding: pedestrian=green, people=yellow
        label = "Pedestrian" if int(cls) == 0 else "People"
        color = (0, 255, 0) if int(cls) == 0 else (0, 200, 255)
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label with background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_frame, (int(x1), int(y1) - th - 10), 
                     (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(vis_frame, text, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(vis_frame, (cx, cy), 3, (0, 0, 255), -1)
    
    return vis_frame


def add_info_overlay(frame, seq_id, frame_name, num_persons, conf_thresh):
    """
    Add information overlay to frame.
    
    Args:
        frame: Input frame
        seq_id: Sequence ID
        frame_name: Frame filename
        num_persons: Number of detected persons
        conf_thresh: Confidence threshold used
    
    Returns:
        Frame with overlay
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Semi-transparent background
    cv2.rectangle(overlay, (10, 10), (420, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Sequence: {seq_id}", (20, 40), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_name}", (20, 70), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Persons: {num_persons}", (20, 100), font, 0.7, (0, 255, 0), 2)
    
    # Confidence threshold indicator
    cv2.putText(frame, f"Conf: {conf_thresh:.2f}", (320, 100), font, 0.5, (100, 200, 255), 1)
    
    # Instructions at bottom
    cv2.putText(frame, "Q: Quit | N: Next seq", (20, h - 20), 
               font, 0.5, (180, 180, 180), 1)
    
    return frame


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    """Main test execution."""
    try:
        # Validate setup
        validate_setup()
        
        # Initialize detector
        print("[1/3] Initializing YOLOv8s detector...")
        try:
            detector = YOLODetector(
                model_path=ONNX_MODEL,
                conf_threshold=CONF_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
                enable_gpu=True  # Set True if CUDA available
            )
            print("[OK] Detector initialized\n")
        except Exception as e:
            print(f"[ERROR] Failed to initialize detector: {str(e)}")
            sys.exit(1)
        
        # Results storage
        all_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": ONNX_MODEL,
            "config": {
                "conf_threshold": CONF_THRESHOLD, 
                "iou_threshold": IOU_THRESHOLD
            },
            "sequences": {}
        }
        
        print(f"[2/3] Processing {len(TEST_SEQUENCES)} sequences...")
        print("-" * 60)
        
        # Process each sequence
        for seq_idx, seq_id in enumerate(TEST_SEQUENCES):
            print(f"\n[{seq_idx+1}/{len(TEST_SEQUENCES)}] Sequence: {seq_id}")
            
            seq_path = os.path.join(DATASET_DIR, seq_id)
            if not os.path.exists(seq_path):
                print(f"  [WARNING] Not found: {seq_path}")
                continue
            
            # Get frame files
            frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg")])
            if not frame_files:
                print(f"  [WARNING] No frames found")
                continue
            
            if MAX_FRAMES_PER_SEQ:
                frame_files = frame_files[:MAX_FRAMES_PER_SEQ]
            
            print(f"  Processing {len(frame_files)} frames...")
            
            # Create sequence output directory
            if SAVE_FRAMES:
                seq_output_dir = os.path.join(OUTPUT_DIR, seq_id)
                os.makedirs(seq_output_dir, exist_ok=True)
            
            # Statistics tracking
            frame_stats = []
            frame_results = {}
            
            # Process frames
            for idx, frame_name in enumerate(frame_files):
                frame_path = os.path.join(seq_path, frame_name)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"    [!] Skipped unreadable: {frame_name}")
                    continue
                
                try:
                    # Detect persons
                    centers, detections = detector.detect_persons_with_boxes(frame)
                    num_people = len(centers)
                    frame_stats.append(num_people)
                    
                    # Store results
                    frame_results[frame_name] = {
                        "num_persons": num_people,
                        "detections": detections.tolist() if len(detections) > 0 else []
                    }
                    
                    print(f"    [{idx+1:2d}/{len(frame_files):2d}] {frame_name}: {num_people:2d} persons")
                    
                    # Visualize
                    vis = draw_detections(frame, detections)
                    vis = add_info_overlay(vis, seq_id, frame_name, num_people, CONF_THRESHOLD)
                    
                    # Save visualization
                    if SAVE_FRAMES:
                        output_path = os.path.join(seq_output_dir, f"det_{frame_name}")
                        cv2.imwrite(output_path, vis)
                    
                    # Display
                    if DISPLAY_FRAMES:
                        cv2.imshow(f"VisDrone Test - {seq_id}", vis)
                        key = cv2.waitKey(200)  # 200ms delay
                        
                        if key & 0xFF == ord('q'):
                            print("\n  [INFO] User quit")
                            cv2.destroyAllWindows()
                            raise KeyboardInterrupt
                        elif key & 0xFF == ord('n'):
                            print("  [INFO] Skipping to next sequence")
                            break
                
                except Exception as e:
                    print(f"    [ERROR] Failed to process {frame_name}: {str(e)}")
                    continue
            
            # Sequence-level statistics
            if frame_stats:
                stats = {
                    "frames_processed": len(frame_stats),
                    "total_frames": len(frame_files),
                    "avg_people": float(np.mean(frame_stats)),
                    "max_people": int(np.max(frame_stats)),
                    "min_people": int(np.min(frame_stats)),
                    "std_people": float(np.std(frame_stats))
                }
                
                all_results["sequences"][seq_id] = {
                    "stats": stats,
                    "frames": frame_results
                }
                
                print(f"\n  Summary:")
                print(f"    Avg persons/frame: {stats['avg_people']:.2f}")
                print(f"    Range: [{stats['min_people']}, {stats['max_people']}]")
        
        # Close display windows
        if DISPLAY_FRAMES:
            cv2.destroyAllWindows()
        
        # Save detection results
        print(f"\n[3/3] Saving results...")
        if SAVE_DETECTIONS and all_results["sequences"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(OUTPUT_DIR, f"detections_{timestamp}.json")
            
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            
            print(f"[OK] Results: {results_path}")
        
        # Print performance summary
        perf = detector.get_stats()
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Frames processed: {perf['frames_processed']}")
        print(f"Avg inference time: {perf['avg_inference_time_ms']:.2f} ms")
        print(f"Avg FPS: {perf['fps']:.2f}")
        
        # Per-sequence results
        if all_results["sequences"]:
            print("\nPER-SEQUENCE RESULTS:")
            print("-" * 60)
            for seq_id, data in all_results["sequences"].items():
                stats = data["stats"]
                print(f"\n{seq_id}:")
                print(f"  Frames: {stats['frames_processed']}/{stats['total_frames']}")
                print(f"  Avg persons: {stats['avg_people']:.2f}")
                print(f"  Range: [{stats['min_people']}, {stats['max_people']}]")
        
        print("\n" + "=" * 60)
        print("[DONE] Test completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)


if __name__ == "__main__":
    main()
