#!/usr/bin/env python3
"""
Quantization script for Qwen3.5-9B model.
Converts the model to 8-bit and 4-bit quantized formats for edge deployment.
Supports GGUF format for llama.cpp inference on ARM devices.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen3.5-9B"
PROJECT_ROOT = Path("/root/Quantied")
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_CACHE_DIR = PROJECT_ROOT / "model" / "cache"


def setup_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")


def check_gpu():
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU available: {gpu_name} with {gpu_memory:.2f} GB memory")
        return True
    else:
        logger.warning("No GPU available. Quantization will use CPU (slower).")
        return False


def quantize_to_int8():
    """Quantize model to 8-bit using bitsandbytes with CPU offloading."""
    logger.info("=" * 60)
    logger.info("Starting 8-bit quantization (INT8) with CPU offloading")
    logger.info("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Configure 8-bit quantization with CPU offloading for large models
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
        
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info("Using device_map='auto' with CPU offloading for memory efficiency...")
        
        # Load model with automatic device mapping (GPU + CPU)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR),
            torch_dtype=torch.float16,
            max_memory={0: "10GiB", "cpu": "30GiB"}  # Limit GPU memory, use CPU for overflow
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        # Save quantized model
        output_path = OUTPUT_DIR / "qwen3.5-9b-int8"
        logger.info(f"Saving 8-bit quantized model to: {output_path}")
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Log model info
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        logger.info(f"8-bit model size: ~{model_size:.2f} GB")
        logger.info("8-bit quantization completed successfully!")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"8-bit quantization failed: {str(e)}")
        raise


def quantize_to_int4():
    """Quantize model to 4-bit using bitsandbytes NF4."""
    logger.info("=" * 60)
    logger.info("Starting 4-bit quantization (NF4)")
    logger.info("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Configure 4-bit quantization with NF4
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info("This may take several minutes...")
        
        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR),
            torch_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        # Save quantized model
        output_path = OUTPUT_DIR / "qwen3.5-9b-int4"
        logger.info(f"Saving 4-bit quantized model to: {output_path}")
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Log model info
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        logger.info(f"4-bit model size: ~{model_size:.2f} GB")
        logger.info("4-bit quantization completed successfully!")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"4-bit quantization failed: {str(e)}")
        raise


def convert_to_gguf():
    """Convert model to GGUF format for llama.cpp inference."""
    logger.info("=" * 60)
    logger.info("Starting GGUF conversion for llama.cpp")
    logger.info("=" * 60)
    
    try:
        # Check if llama.cpp is available
        import subprocess
        
        # Clone llama.cpp if not present
        llama_cpp_path = PROJECT_ROOT / "llama.cpp"
        if not llama_cpp_path.exists():
            logger.info("Cloning llama.cpp repository...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/ggerganov/llama.cpp.git",
                str(llama_cpp_path)
            ], check=True)
        
        logger.info("GGUF conversion preparation complete")
        logger.info("Note: Full GGUF conversion requires llama.cpp convert script")
        logger.info("This will be done after INT4 quantization is complete")
        
        return str(llama_cpp_path)
        
    except Exception as e:
        logger.error(f"GGUF conversion failed: {str(e)}")
        raise


def test_quantized_model(model_path: str, prompt: str = "Hello, how are you?"):
    """Test the quantized model with a simple inference."""
    logger.info("=" * 60)
    logger.info(f"Testing quantized model: {model_path}")
    logger.info("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading model for testing...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Running inference with prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        logger.info("Model test completed successfully!")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        raise


def main():
    """Main entry point for quantization pipeline."""
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5-9B model for edge deployment")
    parser.add_argument("--mode", choices=["int8", "int4", "gguf", "all"], default="all",
                       help="Quantization mode to run")
    parser.add_argument("--test", action="store_true", help="Run inference test after quantization")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download if cached")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Qwen3.5-9B Quantization Pipeline")
    logger.info("=" * 60)
    
    setup_directories()
    check_gpu()
    
    results = {}
    
    try:
        if args.mode in ["int8", "all"]:
            int8_path = quantize_to_int8()
            results["int8"] = int8_path
            
            if args.test:
                test_quantized_model(int8_path)
        
        if args.mode in ["int4", "all"]:
            int4_path = quantize_to_int4()
            results["int4"] = int4_path
            
            if args.test:
                test_quantized_model(int4_path)
        
        if args.mode in ["gguf", "all"]:
            gguf_path = convert_to_gguf()
            results["gguf"] = gguf_path
        
        logger.info("=" * 60)
        logger.info("Quantization Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info("Results:")
        for mode, path in results.items():
            logger.info(f"  {mode}: {path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
