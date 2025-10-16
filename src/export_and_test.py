import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback
from xml.parsers.expat import model

import torch
import numpy as np
import onnx
import onnxruntime as ort
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from tqdm import tqdm

from preprocess.dataloader import CrowdGraphDataset, collate_fn
from models.stgnn import STGNN

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('export_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_PATH = "D:/stgnn_project/src/checkpoints/stgnn_best.pth"
OUTPUT_DIR = "D:/stgnn_project/outputs/evaluation"
DATA_ROOT = "D:/stgnn_project/data/processed/txt_graphs"
VAL_SPLIT = "val"

# Model configuration
MODEL_CONFIG = {
    "in_channels": 5,
    "hidden_channels": 64,
    "out_channels": 2,
    "num_layers": 3
}

# Evaluation configuration
EVAL_CONFIG = {
    "batch_size": 1,
    "min_nodes": 1,  # Minimum nodes required per graph
    "anomaly_threshold_std": 1.0,  # Std deviations above mean for anomaly
    "onnx_opset_version": 17,
    "onnx_test_tolerance": 1e-5
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_paths():
    """Validate all required paths exist."""
    logger.info("Validating paths...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
    
    val_path = os.path.join(DATA_ROOT, VAL_SPLIT)
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation split not found: {val_path}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("[ok] All paths validated successfully")


def validate_checkpoint(checkpoint):
    """Validate checkpoint structure."""
    required_keys = ["model_state_dict"]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Missing required key in checkpoint: {key}")
    
    logger.info("[ok] Checkpoint structure validated")


def validate_graph_data(inp, target):
    """Validate graph data before processing."""
    # Check if data exists
    if inp.x is None or target.x is None:
        return False, "Missing node features"
    
    # Check for empty graphs
    if inp.x.shape[0] == 0 or target.x.shape[0] == 0:
        return False, "Empty graph detected"
    
    # Check node count mismatch
    if inp.x.shape[0] != target.x.shape[0]:
        return False, f"Node count mismatch: input={inp.x.shape[0]}, target={target.x.shape[0]}"
    
    # Check minimum node requirement
    if inp.x.shape[0] < EVAL_CONFIG["min_nodes"]:
        return False, f"Insufficient nodes: {inp.x.shape[0]} < {EVAL_CONFIG['min_nodes']}"
    
    # Check feature dimensions
    if inp.x.shape[1] != MODEL_CONFIG["in_channels"]:
        return False, f"Feature dimension mismatch: {inp.x.shape[1]} != {MODEL_CONFIG['in_channels']}"
    
    if target.x.shape[1] < MODEL_CONFIG["out_channels"]:
        return False, f"Target dimension too small: {target.x.shape[1]} < {MODEL_CONFIG['out_channels']}"
    
    # Check for NaN or Inf values
    if torch.isnan(inp.x).any() or torch.isinf(inp.x).any():
        return False, "NaN or Inf detected in input features"
    
    if torch.isnan(target.x).any() or torch.isinf(target.x).any():
        return False, "NaN or Inf detected in target features"
    
    return True, None


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, loader, device):
    """
    Evaluate model with comprehensive error handling and metrics tracking.
    
    Returns:
        dict: Dictionary containing all evaluation metrics and statistics
    """
    logger.info("Starting model evaluation...")
    model.eval()
    
    all_true, all_pred = [], []
    all_anomaly_scores = []
    
    # Statistics tracking
    stats = {
        "total_sequences": 0,
        "total_frames": 0,
        "skipped_frames": 0,
        "successful_frames": 0,
        "errors": []
    }
    
    with torch.no_grad():
        for seq_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            stats["total_sequences"] += 1
            
            for frame_idx, (inp, target) in enumerate(batch):
                stats["total_frames"] += 1
                
                # Validate graph data
                is_valid, error_msg = validate_graph_data(inp, target)
                if not is_valid:
                    stats["skipped_frames"] += 1
                    stats["errors"].append({
                        "seq": seq_idx,
                        "frame": frame_idx,
                        "reason": error_msg
                    })
                    logger.debug(f"Skipping seq={seq_idx}, frame={frame_idx}: {error_msg}")
                    continue
                
                try:
                    # Move data to device
                    x_in = inp.x.unsqueeze(0).unsqueeze(0).to(device)
                    edge_index = inp.edge_index.to(device)
                    
                    # Forward pass
                    y_pred = model(x_in, edge_index).squeeze(0).cpu().numpy()
                    y_true = target.x[:, :MODEL_CONFIG["out_channels"]].cpu().numpy()
                    
                    # Validate output
                    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                        stats["skipped_frames"] += 1
                        stats["errors"].append({
                            "seq": seq_idx,
                            "frame": frame_idx,
                            "reason": "NaN or Inf in predictions"
                        })
                        logger.warning(f"Invalid predictions at seq={seq_idx}, frame={frame_idx}")
                        continue
                    
                    # Store results
                    all_true.append(y_true)
                    all_pred.append(y_pred)
                    
                    # Compute anomaly score (MSE per frame)
                    anomaly_score = np.mean((y_pred - y_true) ** 2)
                    all_anomaly_scores.append(anomaly_score)
                    
                    stats["successful_frames"] += 1
                    
                except Exception as e:
                    stats["skipped_frames"] += 1
                    stats["errors"].append({
                        "seq": seq_idx,
                        "frame": frame_idx,
                        "reason": f"Exception: {str(e)}"
                    })
                    logger.error(f"Error processing seq={seq_idx}, frame={frame_idx}: {str(e)}")
                    continue
    
    # Check if we have any valid predictions
    if len(all_true) == 0:
        raise ValueError("No valid predictions generated. Check your data and model.")
    
    logger.info(f"Processed {stats['successful_frames']}/{stats['total_frames']} frames successfully")
    logger.info(f"Skipped {stats['skipped_frames']} frames due to errors")
    
    # Concatenate results
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_true, all_pred, all_anomaly_scores)
    metrics["stats"] = stats
    
    return metrics


def calculate_metrics(all_true, all_pred, all_anomaly_scores):
    """Calculate comprehensive evaluation metrics."""
    logger.info("Calculating evaluation metrics...")
    
    metrics = {}
    
    try:
        # Regression metrics
        metrics["mse"] = mean_squared_error(all_true, all_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(all_true, all_pred)
        metrics["r2"] = r2_score(all_true, all_pred)
        
        # Per-coordinate metrics
        metrics["mae_x"] = mean_absolute_error(all_true[:, 0], all_pred[:, 0])
        metrics["mae_y"] = mean_absolute_error(all_true[:, 1], all_pred[:, 1])
        
        # Anomaly detection metrics
        threshold = np.mean(all_anomaly_scores) + EVAL_CONFIG["anomaly_threshold_std"] * np.std(all_anomaly_scores)
        y_anomaly = np.array(all_anomaly_scores) > threshold
        y_true_labels = np.zeros_like(y_anomaly)  # Assuming all normal for demo
        
        metrics["f1_anomaly"] = f1_score(y_true_labels, y_anomaly, zero_division=1)
        metrics["anomaly_threshold"] = threshold
        metrics["detected_anomalies"] = int(y_anomaly.sum())
        metrics["anomaly_rate"] = float(y_anomaly.mean())
        
        # Distribution statistics
        metrics["pred_mean"] = float(np.mean(all_pred))
        metrics["pred_std"] = float(np.std(all_pred))
        metrics["true_mean"] = float(np.mean(all_true))
        metrics["true_std"] = float(np.std(all_true))
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise
    
    return metrics


# ============================================================================
# MODEL EXPORT & VALIDATION
# ============================================================================

def export_pytorch_model(model, output_path):
    """Export model in PyTorch format."""
    try:
        torch.save(model.state_dict(), output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"[ok] PyTorch model exported: {output_path} ({file_size:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to export PyTorch model: {str(e)}")
        return False


def export_onnx_model(model, output_path, device):
    """Export and validate ONNX model."""
    try:
        logger.info("Exporting ONNX model...")
        
        # Create dummy inputs with realistic dimensions
        dummy_x = torch.randn(1, 1, 100, MODEL_CONFIG["in_channels"]).to(device)
        dummy_edge_index = torch.randint(0, 100, (2, 300)).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_x, dummy_edge_index),
            output_path,
            input_names=["x", "edge_index"],
            output_names=["pred_xy"],
            dynamic_axes={
                "x": {2: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "pred_xy": {1: "num_nodes"}
            },
            opset_version=EVAL_CONFIG["onnx_opset_version"],
            do_constant_folding=True,
            export_params=True
        )
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"[ok] ONNX model exported: {output_path} ({file_size:.2f} MB)")
        
        # Validate ONNX model
        validate_onnx_model(model, output_path, device, dummy_x, dummy_edge_index)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def validate_onnx_model(pytorch_model, onnx_path, device, dummy_x, dummy_edge_index):
    """Validate ONNX model against PyTorch model."""
    try:
        logger.info("Validating ONNX model...")
        
        # Check ONNX model structure
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("[ok] ONNX model structure is valid")
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_x, dummy_edge_index).cpu().numpy()
        
        # Get ONNX Runtime output
        ort_inputs = {
            "x": dummy_x.cpu().numpy(),
            "edge_index": dummy_edge_index.cpu().numpy()
        }
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        logger.info(f"PyTorch vs ONNX - Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        
        if max_diff < EVAL_CONFIG["onnx_test_tolerance"]:
            logger.info("[ok] ONNX model validation passed")
            return True
        else:
            logger.warning(f" ONNX output differs from PyTorch (max diff: {max_diff:.6e})")
            return False
            
    except Exception as e:
        logger.error(f"ONNX validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(metrics, output_dir):
    """Save evaluation results to multiple formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.generic):  # Handles all numpy scalar types
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    try:
        metrics_serializable = convert_to_native(metrics)  # Convert before saving
        with open(json_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        logger.info(f"[ok] Metrics saved to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON metrics: {str(e)}")
    
    # Save as text report
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    try:
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("STGNN MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {CHECKPOINT_PATH}\n\n")
            
            f.write("REGRESSION METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"MSE  : {metrics['mse']:.6f}\n")
            f.write(f"RMSE : {metrics['rmse']:.6f}\n")
            f.write(f"MAE  : {metrics['mae']:.6f}\n")
            f.write(f"MAE-X: {metrics['mae_x']:.6f}\n")
            f.write(f"MAE-Y: {metrics['mae_y']:.6f}\n")
            f.write(f"R2   : {metrics['r2']:.6f}\n\n")
            
            f.write("ANOMALY DETECTION\n")
            f.write("-" * 60 + "\n")
            f.write(f"F1 Score       : {metrics['f1_anomaly']:.4f}\n")
            f.write(f"Threshold      : {metrics['anomaly_threshold']:.6f}\n")
            f.write(f"Detected       : {metrics['detected_anomalies']}\n")
            f.write(f"Anomaly Rate   : {metrics['anomaly_rate']:.4f}\n\n")
            
            f.write("STATISTICS\n")
            f.write("-" * 60 + "\n")
            stats = metrics['stats']
            f.write(f"Total Sequences    : {stats['total_sequences']}\n")
            f.write(f"Total Frames       : {stats['total_frames']}\n")
            f.write(f"Successful Frames  : {stats['successful_frames']}\n")
            f.write(f"Skipped Frames     : {stats['skipped_frames']}\n")
            f.write(f"Success Rate       : {stats['successful_frames']/stats['total_frames']*100:.2f}%\n")
            
        logger.info(f"[ok] Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save text report: {str(e)}")



def print_summary(metrics):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print(" FINAL EVALUATION METRICS")
    print("=" * 60)
    print(f"MSE  : {metrics['mse']:.6f}")
    print(f"RMSE : {metrics['rmse']:.6f}")
    print(f"MAE  : {metrics['mae']:.6f}")
    print(f"R²   : {metrics['r2']:.6f}")
    print(f"F1   : {metrics['f1_anomaly']:.4f}")
    print(f"\nMAE-X: {metrics['mae_x']:.6f}")
    print(f"MAE-Y: {metrics['mae_y']:.6f}")
    print(f"\nAnomalies detected: {metrics['detected_anomalies']}")
    print("=" * 60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive error handling."""
    try:
        logger.info("=" * 60)
        logger.info("Starting STGNN Export and Evaluation Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Validate paths
        validate_paths()
        
        # Step 2: Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Step 3: Load model
        logger.info(f"Loading model from: {CHECKPOINT_PATH}")
        model = STGNN(**MODEL_CONFIG).to(device)
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        validate_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("[ok] Model loaded successfully")

        # Step 4: Load dataset
        logger.info("[ok] Loading validation dataset...")
        dataset = CrowdGraphDataset(DATA_ROOT, split=VAL_SPLIT)
        loader = DataLoader(
            dataset,
            batch_size=EVAL_CONFIG["batch_size"],
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues
        )
        logger.info(f"[ok] Loaded {len(dataset)} sequences")
        
        # Step 5: Evaluate model
        metrics = evaluate_model(model, loader, device)
        
        # Step 6: Export models
        logger.info("\nExporting models...")
        pytorch_path = os.path.join(OUTPUT_DIR, "stgnn_final_frozen.pth")
        onnx_path = os.path.join(OUTPUT_DIR, "stgnn_final.onnx")
        
        pytorch_success = export_pytorch_model(model, pytorch_path)
        onnx_success = export_onnx_model(model, onnx_path, device)
        
        # Step 7: Save results
        save_results(metrics, OUTPUT_DIR)
        print_summary(metrics)
        
        # Step 8: Final status
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"PyTorch export: {'[ok]' if pytorch_success else '✗'}")
        logger.info(f"ONNX export: {'[ok]' if onnx_success else '✗'}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
