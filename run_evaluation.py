#!/usr/bin/env python3
"""
Comprehensive evaluation script for quantized Qwen3.5-2B models.
Runs perplexity evaluation and generates comparison report.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/root/Quantied")
OUTPUT_DIR = PROJECT_ROOT / "output"
LLAMA_CPP_BIN = PROJECT_ROOT / "llama.cpp" / "build" / "bin"

def get_model_size(model_path):
    """Get model size in MB."""
    if not model_path.exists():
        return 0
    return model_path.stat().st_size / (1024 * 1024)

def create_test_text():
    """Create a test text file for perplexity evaluation."""
    test_file = OUTPUT_DIR / "test_text.txt"
    test_content = """The capital of France is Paris. It is known for the Eiffel Tower.
London is the capital of the United Kingdom. Berlin is the capital of Germany.
Rome is the capital of Italy. Madrid is the capital of Spain.
Washington D.C. is the capital of the United States. Ottawa is the capital of Canada.
Canberra is the capital of Australia. Tokyo is the capital of Japan.
Beijing is the capital of China. Moscow is the capital of Russia.
Brasília is the capital of Brazil. New Delhi is the capital of India.
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing enables computers to understand human language.
Computer vision allows machines to interpret visual information.
"""
    test_file.write_text(test_content)
    return test_file

def run_perplexity(model_path, test_file, chunks=5):
    """Run perplexity evaluation on a model."""
    logger.info(f"Running perplexity on {model_path.name}...")
    
    cmd = [
        str(LLAMA_CPP_BIN / "llama-perplexity"),
        "-m", str(model_path),
        "-f", str(test_file),
        "-t", "6",
        "--chunks", str(chunks)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Extract perplexity value
        ppl = None
        for line in output.split('\n'):
            if 'perplexity' in line.lower() and ':' in line:
                try:
                    ppl_str = line.split(':')[-1].strip().split()[0]
                    ppl = float(ppl_str)
                except:
                    pass
        
        return {
            'model': model_path.name,
            'size_mb': get_model_size(model_path),
            'perplexity': ppl,
            'success': result.returncode == 0,
            'output': output
        }
    except Exception as e:
        logger.error(f"Error running perplexity: {e}")
        return {
            'model': model_path.name,
            'size_mb': get_model_size(model_path),
            'perplexity': None,
            'success': False,
            'error': str(e)
        }

def test_inference(model_path, prompt="The capital of France is", n_predict=20):
    """Test basic inference on a model."""
    logger.info(f"Testing inference on {model_path.name}...")
    
    cmd = [
        str(LLAMA_CPP_BIN / "llama-cli"),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(n_predict),
        "-t", "6",
        "--temp", "0.7",
        "--no-display-prompt"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Extract generated text
        generated = output.split(prompt)[-1].strip() if prompt in output else output.strip()
        
        return {
            'model': model_path.name,
            'success': result.returncode == 0 and len(generated) > 5,
            'generated': generated[:200] if generated else "N/A",
            'output': output
        }
    except Exception as e:
        logger.error(f"Error testing inference: {e}")
        return {
            'model': model_path.name,
            'success': False,
            'error': str(e)
        }

def main():
    """Main evaluation function."""
    logger.info("Starting comprehensive evaluation...")
    
    # Remove corrupted IQ3_XXS file if exists
    iq3_xxs_path = OUTPUT_DIR / "qwen3.5-2b-IQ3_XXS.gguf"
    if iq3_xxs_path.exists():
        logger.info(f"Removing corrupted file: {iq3_xxs_path}")
        iq3_xxs_path.unlink()
    
    # Create test file
    test_file = create_test_text()
    logger.info(f"Created test file: {test_file}")
    
    # Models to evaluate
    models = [
        OUTPUT_DIR / "qwen3.5-2b-Q4_K_S.gguf",
        OUTPUT_DIR / "qwen3.5-2b-Q3_K_S.gguf",
        OUTPUT_DIR / "qwen3.5-2b-Q2_K.gguf",
    ]
    
    results = []
    
    for model_path in models:
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_path.name}")
        logger.info(f"Size: {get_model_size(model_path):.2f} MB")
        logger.info(f"{'='*60}")
        
        # Run perplexity
        ppl_result = run_perplexity(model_path, test_file, chunks=5)
        
        # Test inference
        inf_result = test_inference(model_path)
        
        results.append({
            'model': model_path.name,
            'size_mb': get_model_size(model_path),
            'perplexity': ppl_result.get('perplexity'),
            'inference_success': inf_result.get('success'),
            'sample_output': inf_result.get('generated', 'N/A')
        })
    
    # Generate report
    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    
    for r in results:
        logger.info(f"\nModel: {r['model']}")
        logger.info(f"  Size: {r['size_mb']:.2f} MB")
        logger.info(f"  Perplexity: {r['perplexity']:.4f}" if r['perplexity'] else "  Perplexity: N/A")
        logger.info(f"  Inference: {'✓ PASS' if r['inference_success'] else '✗ FAIL'}")
    
    logger.info(f"\nReport saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    main()
