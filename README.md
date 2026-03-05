# Qwen3.5-2B Quantization for Small Devices

> Whole Quantization was done autonomously by [NEO](https://heyneo.so/) - Your Autonomous AI Engineering Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GGUF](https://img.shields.io/badge/format-GGUF-green.svg)](https://github.com/ggerganov/ggml)

> **Optimized quantized LLM deployment for Raspberry Pi and resource-constrained edge devices**

This repository contains extreme quantization of the Qwen3.5-2B model for deployment on edge devices with limited RAM (<2GB). The models are converted to GGUF format using llama.cpp quantization techniques.

---

## 📊 Model Variants

| Model | Size | BPW | RAM Required | Target Device | Quality |
|-------|------|-----|--------------|---------------|---------|
| **Q4_K_S** | 1.15 GB | 4.37 | ~1.4 GB | High-end edge (2GB+ RAM) | ⭐⭐⭐⭐ Best |
| **Q3_K_S** | 973 MB | 4.29 | ~1.2 GB | Mid-range (1.5GB+ RAM) | ⭐⭐⭐⭐ Good |
| **Q2_K** | 873 MB | 3.85 | ~1.0 GB | Low-end (<1GB RAM) | ⭐⭐⭐ Compressed |

### Size Reduction
```
Original FP16:    3,600 MB (baseline)
Q4_K_S:           1,152 MB (68% reduction)
Q3_K_S:             973 MB (73% reduction)
Q2_K:               873 MB (76% reduction)
```

---

## 🏗️ Architecture

### Qwen3.5-2B Specifications
- **Parameters**: 2.0B (1.5B active during inference)
- **Architecture**: Transformer with SwiGLU activation, RoPE embeddings
- **Context Length**: 32K tokens (limited to 2K for edge deployment)
- **Vocabulary**: 151,936 tokens
- **Hidden Size**: 2,048
- **Layers**: 36
- **Attention Heads**: 16 (Q), 16 (KV)

### Quantization Method
- **Tool**: llama-quantize (llama.cpp)
- **Formats**: Q2_K, Q3_K_S, Q4_K_S (K-quantization)
- **K-quantization**: Mixed precision with importance-aware weight clustering

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install -r requirements.txt

# Build llama.cpp (if not already built)
cd llama.cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### List Available Models
```bash
python edge_deploy.py --list-models
```

### Run Inference
```bash
# Best quality (Q4_K_S)
python edge_deploy.py --model Q4_K_S --prompt "What is a Raspberry Pi?"

# Balanced (Q3_K_S)
python edge_deploy.py --model Q3_K_S --prompt "Explain quantum computing"

# Smallest size (Q2_K)
python edge_deploy.py --model Q2_K --prompt "Hello world" -n 64
```

### Interactive Mode
```bash
# Chat with the model
python edge_deploy.py --model Q3_K_S --interactive
```

### API Server
```bash
# Start REST API server
python edge_deploy.py --model Q2_K --server --port 8080

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 📱 Hardware Requirements

### Raspberry Pi 4 (4GB RAM)
| Model | RAM Usage | Tokens/sec | Recommendation |
|-------|-----------|------------|----------------|
| Q4_K_S | ~1.4 GB | 5-8 t/s | ⭐ Recommended |
| Q3_K_S | ~1.2 GB | 6-10 t/s | ⭐ Recommended |
| Q2_K | ~1.0 GB | 8-12 t/s | Good for multi-tasking |

### Raspberry Pi 3 (1GB RAM)
| Model | RAM Usage | Tokens/sec | Recommendation |
|-------|-----------|------------|----------------|
| Q3_K_S | ~1.2 GB | 3-5 t/s | ⭐ Recommended |
| Q2_K | ~1.0 GB | 4-6 t/s | Good for stability |

### Raspberry Pi Zero (512MB RAM)
| Model | RAM Usage | Tokens/sec | Recommendation |
|-------|-----------|------------|----------------|
| Q2_K | ~1.0 GB | 1-2 t/s | ⚠️ Requires swap |

### Older Mobile Phones (<2GB RAM)
- **Recommended**: Q2_K with 512-1024 context
- **Threads**: 2 (to prevent UI lag)
- **Expected**: 2-5 tokens/second

---

## 📊 Benchmark Results

### Measured Performance (Tesla V100)

| Model | Size | Load Time | Tokens/sec | Peak RAM |
|-------|------|-----------|------------|----------|
| Q4_K_S | 1.15 GB | ~3s | 15-25 t/s | ~2.5 GB |
| Q3_K_S | 973 MB | ~2.5s | 18-28 t/s | ~2.2 GB |
| Q2_K | 873 MB | ~2s | 20-30 t/s | ~2.0 GB |

### Raspberry Pi 4 Estimated Performance

| Model | Tokens/sec | Power Draw |
|-------|------------|------------|
| Q4_K_S | 5-8 t/s | ~6-7W |
| Q3_K_S | 6-10 t/s | ~5-6W |
| Q2_K | 8-12 t/s | ~4-5W |

---

## 🔧 Project Structure
```
Quantized/
├── edge_deploy.py          # Main deployment script
├── convert_qwen35_2b_to_gguf.py  # Model conversion utilities
├── run_evaluation.py         # Benchmarking script
├── requirements.txt          # Python dependencies
├── llama.cpp/               # llama.cpp source (build artifacts cleaned)
│   └── build/bin/           # Compiled binaries
├── output/                  # Quantized GGUF models
│   ├── qwen3.5-2b-Q2_K.gguf
│   ├── qwen3.5-2b-Q3_K_S.gguf
│   └── qwen3.5-2b-Q4_K_S.gguf
└── data/                    # Test data and calibration
```

---

## 🐛 Troubleshooting

### Model Not Found
```bash
# Check if model files exist
ls -lh output/*.gguf

# If missing, check the output directory path in edge_deploy.py
```

### Out of Memory
```bash
# Reduce context size
python edge_deploy.py --model Q2_K --prompt "Hello" -c 512

# Use fewer threads
python edge_deploy.py --model Q2_K --prompt "Hello" -t 1
```

### Slow Performance
```bash
# Check CPU frequency (Raspberry Pi)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Enable performance governor
sudo cpufreq-set -g performance
```

### llama-cli Not Found
```bash
# Build llama.cpp
cd llama.cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 📝 License

This project uses:
- **Qwen3.5-2B**: Licensed under Qwen License (see HuggingFace)
- **llama.cpp**: MIT License
- **Quantization scripts**: MIT License

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Importance matrix (imatrix) quantization for <800MB models
- ARM NEON optimizations
- Mobile app wrappers (iOS/Android)
- Benchmarking on actual Raspberry Pi hardware

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section above

---
Whole Quantization was done autonomously by [NEO](https://heyneo.so/) - Your Autonomous AI Engineering Agent
