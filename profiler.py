import argparse
import torch
from torch import nn
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling

class RandomSparseGate(nn.Module):
    """Drop-in replacement for l2_gate_proj that generates random binary gates"""
    
    def __init__(self, config, sparsity):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.l2_line_size = config.line_size
        self.intermediate_size = config.intermediate_size
        self.num_blocks = self.intermediate_size // self.l2_line_size
        self.sparsity = sparsity  # fraction of gates that are active
        
    def forward(self, x):
        """Generate random binary gates (0 or 1) based on sparsity probability"""
        batch_size, seq_len, _ = x.shape
        
        # Generate random mask: 1 with probability=sparsity, 0 otherwise
        random_vals = torch.rand(batch_size, seq_len, self.num_blocks,
                                device=x.device, dtype=torch.float32)
        
        gate = (random_vals > self.sparsity)*random_vals
        
        return gate

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size=512):
        """
        dataset: a streaming HF dataset (e.g. load_dataset(..., streaming=True))
        tokenizer: a Hugging Face tokenizer
        block_size: number of tokens per chunk
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
    def __iter__(self):
        buffer_ids = []
        buffer_attn = []

        for example in self.dataset:
            # Tokenize text (or whatever field you want to read)
            tokenized = self.tokenizer(example["text"])
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Accumulate tokens in buffers
            buffer_ids.extend(input_ids)
            buffer_attn.extend(attention_mask)

            # Yield chunks as soon as we have enough tokens
            while len(buffer_ids) >= self.block_size:
                chunk_ids = buffer_ids[: self.block_size]
                chunk_attn = buffer_attn[: self.block_size]
                yield {
                    "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(chunk_attn, dtype=torch.long),
                }
                # Remove used tokens from the buffer
                buffer_ids = buffer_ids[self.block_size :]
                buffer_attn = buffer_attn[self.block_size :]

        # Optional: if you want to yield leftover tokens at the end
        # just comment out if you prefer dropping incomplete chunks
        if buffer_ids:
            yield {
                "input_ids": torch.tensor(buffer_ids, dtype=torch.long),
                "attention_mask": torch.tensor(buffer_attn, dtype=torch.long),
            }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", model_max_length=args.seq_len,padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    train_dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", streaming=True)
    iter_dataset = ChunkedIterableDataset(train_dataset, tokenizer, block_size=args.seq_len)
    config = AutoConfig.from_pretrained(args.config,attn_implementation="sdpa")
    model = AutoModelForCausalLM.from_config(config,attn_implementation="sdpa",torch_dtype=torch.bfloat16).to(device)
    model.train()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable() 

    # Get dummy batch from iter_dataset
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        return_tensors="pt",
    )
    def collate_to_device(batch):
        batch = data_collator(batch)  # creates CPU tensors
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    data_loader = DataLoader(
        iter_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_to_device,
        num_workers=0,
    )
    batch_data = next(iter(data_loader))
    
    if args.random_gates:
        print(f"\n=== Using random binary gates with {args.gate_sparsity:.1%} sparsity ===")
        
        # Create single random gate module and replace all gates
        random_gate = RandomSparseGate(config, args.gate_sparsity).to(device)
        for layer in model.model.layers:
            layer.mlp.l2_gate_proj = random_gate
    
    # Optional warm-up
    for _ in range(args.warmup):
        llm_output = model(**batch_data)
        loss = llm_output.loss + 10**-7*llm_output.l2_reg_loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda": torch.cuda.empty_cache()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=args.profile_memory,
    ) as prof:
        l2_acts = []
        for _ in range(args.iters):
            llm_output = model(**batch_data)
            l2_acts.append(llm_output.l2_act_ratio.mean().item())
            loss = llm_output.loss + 10**-7*llm_output.l2_reg_loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            prof.step()

    if args.random_gates:
        avg_l2_act = sum(l2_acts) / len(l2_acts)
        print(f"\nActual L2 activation: {avg_l2_act:.3f}")

    print("\n=== Top 40 ops by self CPU time ===")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=40))
    if device.type == "cuda":
        print("\n=== Top 40 ops by self CUDA time ===")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile forward & backward of a HuggingFace CausalLM model (text-only output)"
    )
    parser.add_argument("--config")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--iters", type=int, default=10, help="Train iterations to profile")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up iterations (not profiled)")
    parser.add_argument("--profile_memory", action="store_true", help="Profile memory allocation")
    parser.add_argument("--random_gates", action="store_true", help="Use random gates instead of learned gates")
    parser.add_argument("--gate_sparsity", type=float, default=0.1, help="Sparsity for random gates (fraction inactive, e.g., 0.1 = 10%%)")
    args = parser.parse_args()
    main(args)