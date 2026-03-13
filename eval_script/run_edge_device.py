#!/usr/bin/env python
"""
Run Edge Device
---------------
Launch script for a single federated edge device.

Usage:
    python scripts/run_edge_device.py --video <path> [options]
    python scripts/run_edge_device.py --config <yaml_path>

Examples:
    # Run with video file
    python scripts/run_edge_device.py --video data/videos/crowd1.mp4

    # Run with custom device ID
    python scripts/run_edge_device.py --video data/videos/crowd1.mp4 --device-id drone-001

    # Run from config file
    python scripts/run_edge_device.py --config configs/edge_default.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root / "src"))

from federated.edge import EdgeClient, EdgeConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a federated STGNN edge device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to video file or stream URL",
    )
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    input_group.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="Use webcam (default: device 0)",
    )
    
    # Device options
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="Unique device identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        choices=["drone", "raspi", "laptop", "mobile", "server"],
        default="laptop",
        help="Device type category",
    )
    
    # Model paths
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Path to YOLO .pt model",
    )
    parser.add_argument(
        "--stgnn-onnx",
        type=str,
        default=None,
        help="Path to STGNN ONNX model",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable visualization window",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Don't save output video",
    )
    
    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EdgeConfig:
    """Build EdgeConfig from command line arguments."""
    
    # Default paths (project structure)
    project_root = Path(__file__).parent.parent
    default_yolo = project_root / "models" / "yolo11n_person_best.pt"
    default_onnx = project_root / "outputs" / "evaluation" / "stgnn_final.onnx"
    default_output = project_root / "outputs" / "pipeline_results"
    
    # Load from YAML if provided
    if args.config:
        config = EdgeConfig.from_yaml(args.config)
        return config
    
    # Determine video source
    if args.webcam is not None:
        video_source = f"webcam:{args.webcam}"
    else:
        video_source = args.video
    
    # Build config
    return EdgeConfig(
        device_id=args.device_id,
        device_type=args.device_type,
        video_source=video_source,
        yolo_model_path=args.yolo_model or str(default_yolo),
        stgnn_onnx_path=args.stgnn_onnx or str(default_onnx),
        output_dir=args.output_dir or str(default_output),
        display_visualization=not args.no_display,
        save_output_video=not args.no_save_video,
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger("EdgeDevice")
    
    try:
        config = build_config(args)
        logger.info("Configuration loaded")
        logger.info("  Device ID: %s", config.device_id or "(auto)")
        logger.info("  Video: %s", config.video_source)
        logger.info("  YOLO: %s", config.yolo_model_path)
        logger.info("  STGNN: %s", config.stgnn_onnx_path)
        
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    
    # Create and run client
    client = EdgeClient(config)
    
    try:
        logger.info("Starting edge device...")
        client.start(blocking=True)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        
    except Exception as exc:
        logger.error("Runtime error: %s", exc)
        return 1
        
    finally:
        client.stop()
        
        # Print final stats
        stats = client.get_stats()
        logger.info("=== Final Statistics ===")
        logger.info("  Frames processed: %d", stats["frame_count"])
        logger.info("  Average FPS: %.2f", stats["fps"])
        logger.info("  Model version: %d", stats["model_version"])
        
        if stats["training_buffer"]:
            tb_stats = stats["training_buffer"]
            logger.info("  Training samples: %d / %d", 
                       tb_stats["count"], tb_stats["capacity"])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
