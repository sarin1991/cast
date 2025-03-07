import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling
from dataclasses import dataclass, field
from typing import Optional
import transformers
torch.backends.cuda.matmul.allow_tf32=True

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
                    "input_ids": chunk_ids,
                    "attention_mask": chunk_attn,
                }
                # Remove used tokens from the buffer
                buffer_ids = buffer_ids[self.block_size :]
                buffer_attn = buffer_attn[self.block_size :]

        # Optional: if you want to yield leftover tokens at the end
        # just comment out if you prefer dropping incomplete chunks
        if buffer_ids:
            yield {
                "input_ids": buffer_ids,
                "attention_mask": buffer_attn,
            }

@dataclass
class CustomTrainingArguments(TrainingArguments):
    pretrained_model: str = field(default=None)
    config_path: str = field(default=None)
    model_output_path: str = field(default=None)
    max_seq_length: int = field(default=8192)
    response_template: str = field(default="[/INST]")

def main():
    parser = transformers.HfArgumentParser(
        (CustomTrainingArguments)
    )
    (training_args,) = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained_model, model_max_length=training_args.max_seq_length,padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    train_dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", streaming=True)
    iter_dataset = ChunkedIterableDataset(train_dataset, tokenizer, block_size=training_args.per_device_train_batch_size)

    if training_args.config_path:
        config = AutoConfig.from_pretrained(training_args.config_path,attn_implementation="flash_attention_2")
        model = AutoModelForCausalLM.from_config(config,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(training_args.pretrained_model,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    model.to('cuda')
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=iter_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(training_args.model_output_path)

if __name__=="__main__":
    main()