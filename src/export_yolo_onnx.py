"""
Export VisDrone YOLOv8s model to ONNX format for aerial/drone person detection.
Production-ready version with proper error handling and file management.
"""


from ultralytics import YOLO
import os
import sys
import shutil


# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "D:/stgnn_project/models/visdrone-yolov8s.pt"
OUTPUT_DIR = "D:/stgnn_project/src"  # ← Changed to src/ where inference code lives
ONNX_NAME = "visdrone-yolov8s.onnx"
IMG_SIZE = 640  # YOLO standard input size


# Export settings
EXPORT_CONFIG = {
    "format": "onnx",
    "imgsz": IMG_SIZE,
    "simplify": True,   # Simplify ONNX graph
    "dynamic": True,    # Dynamic batch size
    "opset": 17,        # ONNX opset version
}


# ==========================================================
# VALIDATION
# ==========================================================
def validate_setup():
    """Validate paths and model availability."""
    print("=" * 60)
    print("VisDrone YOLOv8s ONNX Export")
    print("=" * 60)
    
    errors = []
    
    # Check if .pt model exists
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model not found: {MODEL_PATH}")
        print(f"[ERROR] Model file not found!")
        print(f"Expected location: {MODEL_PATH}")
        print(f"Please download 'best.pt' from HuggingFace and save as:")
        print(f"  D:/stgnn_project/models/visdrone-yolov8s.pt")
    else:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[OK] Model found: {MODEL_PATH}")
        print(f"     Size: {size_mb:.2f} MB")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Output directory: {OUTPUT_DIR}")
    
    if errors:
        print("\n[ERRORS FOUND]")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    return True


# ==========================================================
# EXPORT
# ==========================================================
def export_model():
    """Export YOLO model to ONNX format."""
    
    print(f"\n[1/3] Loading model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"[OK] Model loaded successfully")
        
        # Display model info
        print(f"\nModel Information:")
        print(f"  - Task: {model.task}")
        print(f"  - Classes: {len(model.names)}")
        print(f"  - Class names: {list(model.names.values())[:5]}...")  # Show first 5
        
        # Check for pedestrian class
        if 'pedestrian' in model.names.values():
            ped_id = [k for k, v in model.names.items() if v == 'pedestrian'][0]
            print(f"  - 'pedestrian' class ID: {ped_id}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Export to ONNX
    print(f"\n[2/3] Exporting to ONNX format...")
    print(f"  - Format: ONNX (opset {EXPORT_CONFIG['opset']})")
    print(f"  - Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Simplify: {EXPORT_CONFIG['simplify']}")
    print(f"  - Dynamic: {EXPORT_CONFIG['dynamic']}")
    
    try:
        export_result = model.export(**EXPORT_CONFIG)
        print(f"[OK] Export completed")
        
    except Exception as e:
        print(f"[ERROR] Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Move ONNX file to output directory
    print(f"\n[3/3] Moving ONNX file to output directory...")
    
    # Ultralytics creates ONNX in same directory as .pt file
    source_onnx = MODEL_PATH.replace('.pt', '.onnx')
    dest_onnx = os.path.join(OUTPUT_DIR, ONNX_NAME)
    
    try:
        if os.path.exists(source_onnx):
            shutil.move(source_onnx, dest_onnx)
            print(f"[OK] ONNX moved to: {dest_onnx}")
        else:
            # Check if it was created in current directory
            current_dir_onnx = ONNX_NAME
            if os.path.exists(current_dir_onnx):
                shutil.move(current_dir_onnx, dest_onnx)
                print(f"[OK] ONNX moved to: {dest_onnx}")
            else:
                print(f"[WARNING] Could not find generated ONNX file")
                print(f"  Expected locations:")
                print(f"    - {source_onnx}")
                print(f"    - {current_dir_onnx}")
                dest_onnx = None
    except Exception as e:
        print(f"[ERROR] Failed to move ONNX file: {str(e)}")
        dest_onnx = None
    
    return dest_onnx


# ==========================================================
# POST-EXPORT INSTRUCTIONS
# ==========================================================
def print_instructions(onnx_path):
    """Print next steps after successful export."""
    if not onnx_path:
        print("\n[WARNING] Please manually locate and move the ONNX file to:")
        print(f"  {OUTPUT_DIR}/{ONNX_NAME}")
        return
    
    print("\n" + "=" * 60)
    print("EXPORT SUCCESSFUL!")
    print("=" * 60)
    
    # File info
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX Model:")
    print(f"  Location: {onnx_path}")
    print(f"  Size: {size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("\n1. Update yolo_detector.py:")
    print("   Change class filter for VisDrone:")
    print("   # VisDrone classes: 0=pedestrian, 1=people")
    print("   person_mask = (class_ids == 0) | (class_ids == 1)")
    
    print(f"\n2. Update test_yolo_on_dataset.py:")
    print(f"   ONNX_MODEL = r'{onnx_path}'")
    print("   CONF_THRESHOLD = 0.10  # Lower for small aerial objects")
    
    print("\n3. Test detection:")
    print("   cd D:/stgnn_project/src")
    print("   python test_yolo_on_dataset.py")
    
    print("\n4. If still no detections, try:")
    print("   - Lower confidence: CONF_THRESHOLD = 0.05")
    print("   - Larger input: input_size=(1280, 1280)")
    
    print("\n" + "=" * 60)


# ==========================================================
# MAIN
# ==========================================================
def main():
    """Main execution."""
    try:
        # Validate setup
        validate_setup()
        
        # Export model
        onnx_path = export_model()
        
        # Print instructions
        print_instructions(onnx_path)
        
        print("\n[DONE] Export pipeline completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Export interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
