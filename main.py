import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, TrainerCallback
from dataclasses import dataclass, field
from typing import Optional, Tuple
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
    initial_sparsity_coefficient: float = field(default=1e-8)
    sparsity_coefficient_multiplier: float = field(default=1.2)
    target_sparsity: float = field(default=0.8)
    

class CustomMetricAccumulator:
    def __init__(self):
        self.metric_sum = 0.0
        self.metric_count = 0

    def update(self, value):
        self.metric_sum += value
        self.metric_count += 1

    def average(self):
        if self.metric_count == 0:
            return 0.0
        return self.metric_sum / self.metric_count

    def reset(self):
        self.metric_sum = 0
        self.metric_count = 0


class MetricCallback(TrainerCallback):
    def __init__(self,metric:CustomMetricAccumulator,metric_name: str, precision: int = 4):
        self.sum = 0.0
        self.count = 0
        self.metric = metric
        self.metric_name = metric_name
        self.precision = precision

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log the average for this interval and reset accumulator
        metric = self.metric.average()
        if logs is not None:
            logs[self.metric_name] = round(metric,self.precision)
        self.metric.reset()


class SparseTrainer(Trainer):
    def __init__(
        self,
        *args,
        current_sparsity_metric_acc: CustomMetricAccumulator,
        l1_reg_loss_metric: CustomMetricAccumulator,
        ce_loss_metric: CustomMetricAccumulator,
        sparsity_coefficient_metric: CustomMetricAccumulator,
        initial_sparsity_coefficient: float = 1e-8,
        sparsity_coefficient_multiplier: float = 1.2,
        target_sparsity: float = 0.8,
        **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self.sparsity_coefficient = initial_sparsity_coefficient
        self.sparsity_coefficient_multiplier = sparsity_coefficient_multiplier
        self.target_sparsity = target_sparsity
        self.current_sparsity_metric_acc = current_sparsity_metric_acc
        self.l1_reg_loss_metric = l1_reg_loss_metric
        self.ce_loss_metric = ce_loss_metric
        self.sparsity_coefficient_metric = sparsity_coefficient_metric

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by SparseTrainer.
        """
        outputs = model(**inputs)
        ce_loss = outputs.loss
        self.ce_loss_metric.update(ce_loss.item())
        sparse_ratio_local = outputs.sparse_ratio.mean()
        l1_reg_loss = outputs.l1_reg_loss.mean()

        # global sparse ratio
        current_sparsity = self.accelerator.reduce(
            sparse_ratio_local.to(self.accelerator.device), reduction='mean'
        ).item()
        self.current_sparsity_metric_acc.update(current_sparsity)

        # global l1_reg_loss
        global_l1_reg_loss = self.accelerator.reduce(
            l1_reg_loss.to(self.accelerator.device), reduction='mean'
        ).item()
        self.l1_reg_loss_metric.update(global_l1_reg_loss)

        if current_sparsity<self.target_sparsity:
            self.sparsity_coefficient = self.sparsity_coefficient*self.sparsity_coefficient_multiplier
        else:
            self.sparsity_coefficient = self.sparsity_coefficient/self.sparsity_coefficient_multiplier
        self.sparsity_coefficient_metric.update(self.sparsity_coefficient)

        # total loss
        loss = ce_loss + self.sparsity_coefficient*l1_reg_loss

        return (loss, outputs) if return_outputs else loss

def main():
    parser = transformers.HfArgumentParser(
        (CustomTrainingArguments)
    )
    parsed_vals: Tuple[CustomTrainingArguments,] = parser.parse_args_into_dataclasses()
    (training_args,) = parsed_vals
    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained_model, model_max_length=training_args.max_seq_length,padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    train_dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", streaming=True)
    iter_dataset = ChunkedIterableDataset(train_dataset, tokenizer, block_size=training_args.max_seq_length)

    if training_args.config_path:
        config = AutoConfig.from_pretrained(training_args.config_path,attn_implementation="sdpa")
        model = AutoModelForCausalLM.from_config(config,attn_implementation="sdpa",torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(training_args.pretrained_model,attn_implementation="sdpa",torch_dtype=torch.bfloat16)
    model.to('cuda')
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    current_sparsity_metric_acc = CustomMetricAccumulator()
    l1_reg_loss_metric = CustomMetricAccumulator()
    ce_loss_metric = CustomMetricAccumulator()
    sparsity_coefficient_metric = CustomMetricAccumulator()
    trainer = SparseTrainer(
        initial_sparsity_coefficient = training_args.initial_sparsity_coefficient,
        sparsity_coefficient_multiplier = training_args.sparsity_coefficient_multiplier,
        target_sparsity = training_args.target_sparsity,
        model=model,
        args=training_args,
        train_dataset=iter_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[MetricCallback(current_sparsity_metric_acc,metric_name='sparsity'),
                   MetricCallback(l1_reg_loss_metric,metric_name='l1_reg_loss'),
                   MetricCallback(ce_loss_metric,metric_name='ce_loss'),
                   MetricCallback(sparsity_coefficient_metric,metric_name='sparsity_coefficient',precision=8)],
        current_sparsity_metric_acc=current_sparsity_metric_acc,
        l1_reg_loss_metric=l1_reg_loss_metric,
        ce_loss_metric=ce_loss_metric,
        sparsity_coefficient_metric=sparsity_coefficient_metric,
    )
    trainer.train()
    trainer.save_model(training_args.model_output_path)

if __name__=="__main__":
    main()