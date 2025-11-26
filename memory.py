#!/usr/bin/env python3
"""
Memory requirement estimator for DeepSpeed ZeRO Stages 1, 2, and 3
"""

from transformers import AutoConfig, AutoModelForCausalLM
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch

def main():
    # Load model
    print("Loading model...")
    config = AutoConfig.from_pretrained('config_cast_180m.json',attn_implementation="sdpa")
    model = AutoModelForCausalLM.from_config(config,attn_implementation="sdpa",torch_dtype=torch.bfloat16)

    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B\n")

    # Stage 1 & 2 estimation
    print(f"{'='*80}")
    print("ZeRO Stage 1 & 2 Memory Estimation")
    print(f"{'='*80}")
    estimate_zero2_model_states_mem_needs_all_live(
        model,
        num_gpus_per_node=1,
        num_nodes=1,
        additional_buffer_factor=1.5
    )

    # Stage 3 estimation
    print(f"\n{'='*80}")
    print("ZeRO Stage 3 Memory Estimation")
    print(f"{'='*80}")
    estimate_zero3_model_states_mem_needs_all_live(
        model,
        num_gpus_per_node=1,
        num_nodes=1,
        additional_buffer_factor=1.5
    )

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
