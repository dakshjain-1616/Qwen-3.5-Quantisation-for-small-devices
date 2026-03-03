#!/usr/bin/env python3
"""
Edge Deployment Script for Qwen3.5-2B Quantized Models
Optimized for Raspberry Pi and mobile devices with limited RAM.

Usage:
    python edge_deploy.py --model Q4_K_S --prompt "What is a Raspberry Pi?"
    python edge_deploy.py --model Q2_K --interactive
    python edge_deploy.py --model Q3_K_S --server --port 8080
    python edge_deploy.py --list-models
"""

import os
import sys
import argparse
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path("/root/Quantied")
OUTPUT_DIR = PROJECT_ROOT / "output"
LLAMA_CPP_BIN = PROJECT_ROOT / "llama.cpp" / "build" / "bin"

# Available models with metadata
AVAILABLE_MODELS = {
    "Q4_K_S": {
        "file": "qwen3.5-2b-Q4_K_S.gguf",
        "size_mb": 1151.78,
        "bpw": 4.37,
        "description": "Best quality, largest size (1.2GB)",
        "recommended_for": "Devices with 2GB+ RAM",
        "context_size": 2048,
        "threads": 4
    },
    "Q3_K_S": {
        "file": "qwen3.5-2b-Q3_K_S.gguf",
        "size_mb": 972.91,
        "bpw": 4.29,
        "description": "Good balance of quality and size (973MB)",
        "recommended_for": "Devices with 1.5GB+ RAM",
        "context_size": 2048,
        "threads": 4
    },
    "Q2_K": {
        "file": "qwen3.5-2b-Q2_K.gguf",
        "size_mb": 873.05,
        "bpw": 3.85,
        "description": "Most compressed, smallest size (874MB)",
        "recommended_for": "Devices with <1GB RAM (Raspberry Pi Zero, older mobile)",
        "context_size": 1024,
        "threads": 2
    }
}


def list_models():
    """List all available models with details."""
    print("\n" + "="*70)
    print("AVAILABLE QUANTIZED MODELS")
    print("="*70)
    
    for model_key, info in AVAILABLE_MODELS.items():
        model_path = OUTPUT_DIR / info['file']
        exists = "✓" if model_path.exists() else "✗"
        
        print(f"\n{exists} {model_key}")
        print(f"   File: {info['file']}")
        print(f"   Size: {info['size_mb']:.1f} MB")
        print(f"   Bits per weight: {info['bpw']:.2f}")
        print(f"   Description: {info['description']}")
        print(f"   Recommended for: {info['recommended_for']}")
        print(f"   Context size: {info['context_size']}")
        print(f"   Threads: {info['threads']}")
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES:")
    print("="*70)
    print("  python edge_deploy.py --model Q4_K_S --prompt 'Hello world'")
    print("  python edge_deploy.py --model Q2_K --interactive")
    print("  python edge_deploy.py --model Q3_K_S --server --port 8080")
    print("="*70 + "\n")


def get_model_path(model_key: str) -> Optional[Path]:
    """Get the path to a model file."""
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return None
    
    model_info = AVAILABLE_MODELS[model_key]
    model_path = OUTPUT_DIR / model_info['file']
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    return model_path


def run_inference(model_key: str, prompt: str, max_tokens: int = 128, **kwargs):
    """Run inference with the specified model."""
    model_path = get_model_path(model_key)
    if not model_path:
        return None
    
    model_info = AVAILABLE_MODELS[model_key]
    
    # Build command
    cmd = [
        str(LLAMA_CPP_BIN / "llama-cli"),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(kwargs.get("threads", model_info["threads"])),
        "-c", str(kwargs.get("ctx_size", model_info["context_size"])),
        "--temp", str(kwargs.get("temp", 0.7)),
        "--repeat-penalty", "1.1",
    ]
    
    logger.info(f"Running inference with {model_key}")
    logger.info(f"Prompt: '{prompt[:50]}...'" if len(prompt) > 50 else f"Prompt: '{prompt}'")
    logger.info(f"Config: threads={cmd[cmd.index('-t')+1]}, ctx_size={cmd[cmd.index('-c')+1]}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            output = result.stdout
            logger.info("Inference completed successfully")
            return output
        else:
            logger.error(f"Inference failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Inference timed out")
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def run_interactive(model_key: str):
    """Run interactive chat mode."""
    model_path = get_model_path(model_key)
    if not model_path:
        return
    
    model_info = AVAILABLE_MODELS[model_key]
    
    cmd = [
        str(LLAMA_CPP_BIN / "llama-cli"),
        "-m", str(model_path),
        "-t", str(model_info["threads"]),
        "-c", str(model_info["context_size"]),
        "--temp", "0.7",
        "--repeat-penalty", "1.1",
        "--interactive"
    ]
    
    logger.info(f"Starting interactive mode with {model_key}...")
    logger.info("Type your prompts and press Enter. Use Ctrl+C to exit.")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("\nExiting interactive mode.")


def run_server(model_key: str, port: int = 8080):
    """Run API server mode."""
    model_path = get_model_path(model_key)
    if not model_path:
        return
    
    model_info = AVAILABLE_MODELS[model_key]
    
    cmd = [
        str(LLAMA_CPP_BIN / "llama-server"),
        "-m", str(model_path),
        "-t", str(model_info["threads"]),
        "-c", str(model_info["context_size"]),
        "--port", str(port),
        "--host", "0.0.0.0"
    ]
    
    logger.info(f"Starting API server with {model_key} on port {port}...")
    logger.info(f"API endpoint: http://localhost:{port}/v1/chat/completions")
    logger.info("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("\nStopping server.")


def main():
    parser = argparse.ArgumentParser(
        description="Edge Deployment Script for Qwen3.5-2B Quantized Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available models
    python edge_deploy.py --list-models
    
    # Run inference with specific model
    python edge_deploy.py --model Q4_K_S --prompt "What is a Raspberry Pi?"
    
    # Interactive mode with Q2_K (smallest model)
    python edge_deploy.py --model Q2_K --interactive
    
    # API server with Q3_K_S
    python edge_deploy.py --model Q3_K_S --server --port 8080
        """
    )
    
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--model", type=str, choices=list(AVAILABLE_MODELS.keys()), 
                        help="Model to use (Q2_K, Q3_K_S, or Q4_K_S)")
    parser.add_argument("-p", "--prompt", type=str, help="Prompt for single inference")
    parser.add_argument("-n", "--n-predict", type=int, default=128, help="Number of tokens to predict")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run interactive chat mode")
    parser.add_argument("-s", "--server", action="store_true", help="Run API server mode")
    parser.add_argument("--port", type=int, default=8080, help="Port for API server")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    if not args.model:
        logger.error("Please specify a model with --model or use --list-models to see available options")
        sys.exit(1)
    
    if args.interactive:
        run_interactive(args.model)
    elif args.server:
        run_server(args.model, args.port)
    elif args.prompt:
        result = run_inference(args.model, args.prompt, args.n_predict, temp=args.temp)
        if result:
            print("\n" + "="*60)
            print("OUTPUT:")
            print("="*60)
            print(result)
    else:
        # Default: run a test inference
        logger.info("No mode specified. Running test inference...")
        result = run_inference(args.model, "What is a Raspberry Pi?", 64)
        if result:
            print("\n" + "="*60)
            print("OUTPUT:")
            print("="*60)
            print(result)


if __name__ == "__main__":
    main()
