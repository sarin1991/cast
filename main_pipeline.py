import os
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, TrainerCallback, get_cosine_schedule_with_warmup
from dataclasses import dataclass, field
from typing import Tuple
import transformers
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
torch.backends.cuda.matmul.allow_tf32=True

RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0

class MemoryCallback(TrainerCallback):
    def _stats(self):
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated()/1024**3
        r = torch.cuda.memory_reserved()/1024**3
        return a, r

    def on_step_begin(self, args, state, control, **kwargs):
        self.base_a, self.base_r = self._stats()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.bwd_a, self.bwd_r = self._stats()
        print(f"[step {state.global_step}] base a/r {self.base_a:.2f}/{self.base_r:.2f}G | "
              f"after_bwd a/r {self.bwd_a:.2f}/{self.bwd_r:.2f}G", flush=True)

    def on_optimizer_step(self, args, state, control, **kwargs):
        opt_a, opt_r = self._stats()
        print(f"[step {state.global_step}] after_opt a/r {opt_a:.2f}/{opt_r:.2f}G", flush=True)

def print_memory_stats(state):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    pa = torch.cuda.max_memory_allocated() / 1024**3
    pr = torch.cuda.max_memory_reserved()  / 1024**3
    print(f"[{state}] a/r {a:.2f}/{r:.2f}G | peak a/r {pa:.2f}/{pr:.2f}G", flush=True)

def module_pre_fwd_hook(layer_idx):
    def _pre(module, inputs):
        torch.cuda.reset_peak_memory_stats()
        print_memory_stats(f"fwd pre layer {layer_idx}")
    return _pre

def module_post_fwd_hook(layer_idx):
    def _post(module, inputs, output):
        print_memory_stats(f"fwd post layer {layer_idx}")
    return _post

def module_pre_bwd_hook(layer_idx):
    def _pre(module, grad_output):
        torch.cuda.reset_peak_memory_stats()
        print_memory_stats(f"bwd pre layer {layer_idx}")
    return _pre

def module_post_bwd_hook(layer_idx):
    def _post(module, grad_input, grad_output):
        print_memory_stats(f"bwd post layer {layer_idx}")
    return _post


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
        if RANK != 0:
            dummy_ids = [0] * self.block_size
            dummy_attn = [1] * self.block_size
            while True:
                yield {
                    "input_ids": dummy_ids,
                    "attention_mask": dummy_attn,
                }

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
    print_memory_stats: bool = field(
        default=False,
        metadata={"help": "Print CUDA memory stats during training."}
    )
    pipeline_n_microbatches: int = field(
        default=4,
        metadata={"help": "Number of pipeline microbatches."}
    )

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


class PipelineLossFn:
    def __init__(self,l2_act_metric:CustomMetricAccumulator,ce_loss_metric:CustomMetricAccumulator):
        self.l2_act_metric = l2_act_metric
        self.ce_loss_metric = ce_loss_metric

    def __call__(self, output, target=None):
        loss = output.loss
        self.ce_loss_metric.update(loss.item())
        self.l2_act_metric.update(output.l2_act_ratio.mean().item())
        return loss


class PipelineTrainer(Trainer):
    def __init__(
        self,
        *args,
        pipeline_schedule=None,
        **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self.pipeline_schedule = pipeline_schedule

    def _wrap_model(self, model, training=True, dataloader=None):
        return model

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        if self.args.print_memory_stats:
            torch.cuda.reset_peak_memory_stats()

        if RANK == 0:
            inputs = self._prepare_inputs(inputs)
            self.pipeline_schedule.step(**inputs)
            loss = torch.tensor(0.0, device=self.args.device)
        elif RANK == WORLD_SIZE - 1:
            output = self.pipeline_schedule.step()
            loss = output.loss
        else:
            self.pipeline_schedule.step()
            loss = torch.tensor(0.0, device=self.args.device)

        if self.args.print_memory_stats:
            print_memory_stats(f'after_pipeline_step {self.state.global_step}')

        return loss.detach() / self.args.gradient_accumulation_steps


def init_distributed():
    global RANK, WORLD_SIZE, LOCAL_RANK
    dist.init_process_group(backend="nccl")
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    torch.cuda.set_device(LOCAL_RANK)

def get_stage_range(num_layers, rank, world_size):
    start = (num_layers * rank) // world_size
    end = (num_layers * (rank + 1)) // world_size
    return start, end

def prune_model_for_stage(model):
    layer_keys = sorted(model.model.layers.keys(), key=int)
    start, end = get_stage_range(len(layer_keys), RANK, WORLD_SIZE)
    keep = set(layer_keys[start:end])

    for key in list(model.model.layers.keys()):
        if key not in keep:
            del model.model.layers[key]

    if RANK > 0:
        model.model.embed_tokens = None

    if RANK < WORLD_SIZE - 1:
        model.model.norm = None
        model.lm_head = None

    return model

def main():
    init_distributed()

    parser = transformers.HfArgumentParser(
        (CustomTrainingArguments)
    )
    parsed_vals: Tuple[CustomTrainingArguments,] = parser.parse_args_into_dataclasses()
    (training_args,) = parsed_vals
    training_args.gradient_checkpointing_kwargs={"use_reentrant": True}

    tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained_model, model_max_length=training_args.max_seq_length,padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    train_dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", streaming=True)
    iter_dataset = ChunkedIterableDataset(train_dataset, tokenizer, block_size=training_args.max_seq_length)

    if training_args.config_path:
        config = AutoConfig.from_pretrained(training_args.config_path,attn_implementation="sdpa")
        model = AutoModelForCausalLM.from_config(config,attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(training_args.pretrained_model,attn_implementation="sdpa")

    model = prune_model_for_stage(model)
    model.to(f'cuda:{LOCAL_RANK}')

    if training_args.print_memory_stats:
        for layer_key in sorted(model.model.layers.keys(), key=int):
            decoder_layer = model.model.layers[layer_key]
            layer_idx = int(layer_key)
            decoder_layer.register_forward_pre_hook(module_pre_fwd_hook(layer_idx))
            decoder_layer.register_forward_hook(module_post_fwd_hook(layer_idx))
            decoder_layer.register_full_backward_pre_hook(module_pre_bwd_hook(layer_idx))
            decoder_layer.register_full_backward_hook(module_post_bwd_hook(layer_idx))
            decoder_layer.mlp.register_forward_pre_hook(module_pre_fwd_hook(f'mlp_{layer_idx}'))
            decoder_layer.mlp.register_forward_hook(module_post_fwd_hook(f'mlp_{layer_idx}'))
            decoder_layer.mlp.register_full_backward_pre_hook(module_pre_bwd_hook(f'mlp_{layer_idx}'))
            decoder_layer.mlp.register_full_backward_hook(module_post_bwd_hook(f'mlp_{layer_idx}'))
            decoder_layer.self_attn.register_forward_pre_hook(module_pre_fwd_hook(f'attn_{layer_idx}'))
            decoder_layer.self_attn.register_forward_hook(module_post_fwd_hook(f'attn_{layer_idx}'))
            decoder_layer.self_attn.register_full_backward_pre_hook(module_pre_bwd_hook(f'attn_{layer_idx}'))
            decoder_layer.self_attn.register_full_backward_hook(module_post_bwd_hook(f'attn_{layer_idx}'))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
    )
    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )
    optimizers=(optimizer, cosine_scheduler)

    l2_act_metric = CustomMetricAccumulator()
    ce_loss_metric = CustomMetricAccumulator()
    pipeline_loss_fn = PipelineLossFn(l2_act_metric,ce_loss_metric)

    stage = PipelineStage(
        model,
        RANK,
        WORLD_SIZE,
        torch.device(f"cuda:{LOCAL_RANK}")
    )
    schedule = ScheduleGPipe(
        stage,
        n_microbatches=training_args.pipeline_n_microbatches,
        loss_fn=pipeline_loss_fn
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    callbacks = []
    if RANK == WORLD_SIZE - 1:
        callbacks.extend([
            MetricCallback(l2_act_metric,metric_name='l2_act'),
            MetricCallback(ce_loss_metric,metric_name='ce_loss'),
        ])
    if training_args.print_memory_stats:
        callbacks.append(MemoryCallback())
    trainer = PipelineTrainer(
        model=model,
        args=training_args,
        train_dataset=iter_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=optimizers,
        pipeline_schedule=schedule,
    )
    trainer.train()

if __name__=="__main__":
    main()