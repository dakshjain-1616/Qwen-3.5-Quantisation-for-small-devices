#!/usr/bin/env python3
"""
Convert Qwen3.5-9B HuggingFace model to GGUF FP16 format.
This script properly handles the conversion for llama.cpp quantization.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/root/Quantied")
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
OUTPUT_DIR = PROJECT_ROOT / "output"
HF_MODEL_PATH = OUTPUT_DIR / "qwen3.5-9b-hf"

def convert_to_gguf():
    """Convert HF model to GGUF FP16 format."""
    logger.info("=" * 60)
    logger.info("Converting Qwen3.5-9B to GGUF FP16 format")
    logger.info("=" * 60)
    
    if not HF_MODEL_PATH.exists():
        logger.error(f"HF model not found at {HF_MODEL_PATH}")
        return False
    
    output_file = OUTPUT_DIR / "qwen3.5-9b-f16.gguf"
    
    # Use the convert_hf_to_gguf.py script
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        logger.error(f"Convert script not found at {convert_script}")
        return False
    
    logger.info(f"Source: {HF_MODEL_PATH}")
    logger.info(f"Output: {output_file}")
    
    # Build command
    cmd = [
        sys.executable, str(convert_script),
        str(HF_MODEL_PATH),
        "--outfile", str(output_file),
        "--outtype", "f16"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info("Conversion completed successfully!")
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"Output file size: {size_mb:.2f} MB")
            return True
        else:
            logger.error(f"Conversion failed with return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return False


if __name__ == "__main__":
    success = convert_to_gguf()
    sys.exit(0 if success else 1)
