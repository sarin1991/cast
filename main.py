import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass, field
from typing import Optional
import transformers
torch.backends.cuda.matmul.allow_tf32=True

@dataclass
class CustomTrainingArguments(SFTConfig):
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
    if training_args.config_path:
        config = AutoConfig.from_pretrained(training_args.config_path,attn_implementation="flash_attention_2")
        model = AutoModelForCausalLM.from_config(config,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(training_args.pretrained_model,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    model.to('cuda')
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.model_output_path)

if __name__=="__main__":
    main()