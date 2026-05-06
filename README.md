# LLM Fine-Tuning with LoRA on Apple Silicon

Fine-tuning a 1B parameter instruction-following language model using Low-Rank Adaptation (LoRA) on consumer hardware (Apple M1, 8GB unified memory). Built as a clean, configurable pipeline designed for reproducibility and easy model swapping.

## Overview

This project demonstrates parameter-efficient fine-tuning (PEFT) of a causal language model on the [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) instruction dataset. Training is done entirely on-device using PyTorch MPS backend — no cloud GPU required.

**Model:** `meta-llama/Llama-3.2-1B-Instruct`  
**Technique:** LoRA (r=16, alpha=32) — trains ~0.5% of total parameters  
**Dataset:** Dolly-15k (5,000 samples, 95/5 train/val split)  
**Hardware:** Apple M1 8GB — ~2.5 hours per epoch  

## Requirements

- Python 3.10+
- PyTorch 2.1+ with MPS backend
- Apple Silicon Mac (M1/M2/M3) or CUDA GPU
- 8GB RAM minimum
- HuggingFace account with Meta Llama access approved