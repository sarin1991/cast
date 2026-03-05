import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, TrainerCallback, get_cosine_schedule_with_warmup
from dataclasses import dataclass, field
from typing import Tuple
import transformers
import json
import deepspeed
from optimizers.layerwise_optim_factory import create_layerwise_optimizer
from optimizers.hooks import attach_multi_param_opt_hook, attach_clear_grad_after_accumulate # Update usages to new modules as needed
torch.backends.cuda.matmul.allow_tf32=True

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

def get_zero_stage(deepspeed_config_path):
    if deepspeed_config_path is None:
        return 0
    with open(deepspeed_config_path, 'r') as f:
        ds_config = json.load(f)
    return ds_config.get("zero_optimization", {}).get("stage", 0)

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
    print_memory_stats: bool = field(
        default=False,
        metadata={"help": "Print CUDA memory stats during training."}
    )
    layerwise_optim: str = field(
        default="none",
        metadata={"help": "Hook-based layerwise optimizer to use: none, layerwise_low_rank_muon, ..."}
    )
    layerwise_optim_kwargs: str = field(
        default="{}",
        metadata={"help": "JSON dict of kwargs for the hook optimizer, e.g. '{\"rank\":8}'"}
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


class SparseTrainer(Trainer):
    def __init__(
        self,
        *args,
        l2_act_metric: CustomMetricAccumulator,
        ce_loss_metric: CustomMetricAccumulator,
        **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self.l2_act_metric = l2_act_metric
        self.ce_loss_metric = ce_loss_metric

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by SparseTrainer.
        """
        if self.args.print_memory_stats:
            torch.cuda.reset_peak_memory_stats()

        outputs = model(**inputs)
        if self.args.print_memory_stats:
            print_memory_stats(f'after_fwd {self.state.global_step}')
        
        ce_loss = outputs.loss
        self.ce_loss_metric.update(ce_loss.item())
        l2_act_local = outputs.l2_act_ratio.mean()

        # global act ratio
        l2_act = self.accelerator.reduce(
            l2_act_local.to(self.accelerator.device), reduction='mean'
        ).item()
        self.l2_act_metric.update(l2_act)

        # total loss
        loss = ce_loss

        return (loss, outputs) if return_outputs else loss

def main():
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

    zero_stage = get_zero_stage(training_args.deepspeed)
    optimizers = (None, None)
    # Optimized model initialization for DeepSpeed ZeRO Stage 3
    if zero_stage == 3:
        # Read the full config and extract only what zero.Init needs
        with open(training_args.deepspeed, 'r') as f:
            full_config = json.load(f)
        
        # Create minimal config for zero.Init with required batch size
        zero_init_config = {
            "train_micro_batch_size_per_gpu": 1,  # Dummy value, Trainer will override
            "zero_optimization": full_config.get("zero_optimization", {}),
            "bf16": full_config.get("bf16", {}),
            "fp16": full_config.get("fp16", {})
        }
        
        # Use config_dict_or_path parameter (not deprecated 'config')
        with deepspeed.zero.Init(config_dict_or_path=zero_init_config):
            if training_args.config_path:
                config = AutoConfig.from_pretrained(training_args.config_path, attn_implementation="sdpa")
                model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
            else:
                model = AutoModelForCausalLM.from_pretrained(training_args.pretrained_model, attn_implementation="sdpa")
    else:
        # Standard initialization without DeepSpeed
        if training_args.config_path:
            config = AutoConfig.from_pretrained(training_args.config_path,attn_implementation="sdpa")
            model = AutoModelForCausalLM.from_config(config,attn_implementation="sdpa")
        else:
            model = AutoModelForCausalLM.from_pretrained(training_args.pretrained_model,attn_implementation="sdpa")
        model.to('cuda')
    
        # Custom layerwise optimizer
        if training_args.layerwise_optim not in (None, "", "none"):
            # --- Partition Model Parameters (robust: uses param identity, not names) ---
            mlp_params_by_layer = []
            mlp_param_ids = set()

            for layer_idx, decoder_layer in enumerate(model.model.layers):
                mlp_params = []
                for name, param in decoder_layer.named_parameters():
                    if ("up_proj" in name) or ("down_proj" in name):
                        mlp_params.append(param)
                        mlp_param_ids.add(id(param))
                mlp_params_by_layer.append(mlp_params)
                if training_args.print_memory_stats:
                    decoder_layer.register_forward_pre_hook(module_pre_fwd_hook(layer_idx))
                    decoder_layer.register_forward_hook(module_post_fwd_hook(layer_idx))
                    decoder_layer.register_full_backward_pre_hook(module_pre_bwd_hook(layer_idx))
                    decoder_layer.register_full_backward_hook(module_post_bwd_hook(layer_idx))
                    # mlp
                    decoder_layer.mlp.register_forward_pre_hook(module_pre_fwd_hook(f'mlp_{layer_idx}'))
                    decoder_layer.mlp.register_forward_hook(module_post_fwd_hook(f'mlp_{layer_idx}'))
                    decoder_layer.mlp.register_full_backward_pre_hook(module_pre_bwd_hook(f'mlp_{layer_idx}'))
                    decoder_layer.mlp.register_full_backward_hook(module_post_bwd_hook(f'mlp_{layer_idx}'))
                    # attn
                    decoder_layer.self_attn.register_forward_pre_hook(module_pre_fwd_hook(f'attn_{layer_idx}'))
                    decoder_layer.self_attn.register_forward_hook(module_post_fwd_hook(f'attn_{layer_idx}'))
                    decoder_layer.self_attn.register_full_backward_pre_hook(module_pre_bwd_hook(f'attn_{layer_idx}'))
                    decoder_layer.self_attn.register_full_backward_hook(module_post_bwd_hook(f'attn_{layer_idx}'))

            other_params = [
                p for p in model.parameters()
                if p.requires_grad and (id(p) not in mlp_param_ids)
            ]

            print(
                f"Found {sum(len(x) for x in mlp_params_by_layer)} params for layerwise optimizer "
                f"and {len(other_params)} for AdamW."
            )

            # --- Create the Two Optimizers ---
            # Note: You need bitsandbytes installed for the 8-bit optimizer.
            layerwise_optimizers = []
            for layer_idx, mlp_params in enumerate(mlp_params_by_layer):
                if len(mlp_params) != 2:
                    raise RuntimeError(
                        f"Layer {layer_idx}: expected 2 params (up_proj/down_proj), found {len(mlp_params)}. "
                        "Check your name filter or model architecture."
                    )

                param_groups = [
                    {
                        'params': mlp_params,
                    }
                ]

                layerwise_optimizer = create_layerwise_optimizer(
                    name=training_args.layerwise_optim,
                    param_groups=param_groups,
                    lr=training_args.learning_rate,
                    kwargs_json=training_args.layerwise_optim_kwargs,
                )
                attach_multi_param_opt_hook(mlp_params, layerwise_optimizer, training_args.print_memory_stats)
                for param in mlp_params:
                    attach_clear_grad_after_accumulate(param)
                layerwise_optimizers.append(layerwise_optimizer)

            all_mlp_params = []
            for mlp_params in mlp_params_by_layer:
                all_mlp_params.extend(mlp_params)

            adamw_optimizer = torch.optim.AdamW(
                other_params,
                lr=training_args.learning_rate, # Also controlled by the scheduler
            )

            # --- Create the Cosine Scheduler ---
            # The scheduler will control the learning rate for BOTH optimizers.
            cosine_scheduler = get_cosine_schedule_with_warmup(
                optimizer=adamw_optimizer,
                num_warmup_steps=training_args.warmup_steps,
                num_training_steps=training_args.max_steps,
            )
            optimizers=(adamw_optimizer, cosine_scheduler)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    l2_act_metric = CustomMetricAccumulator()
    ce_loss_metric = CustomMetricAccumulator()
    callbacks = [
        MetricCallback(l2_act_metric,metric_name='l2_act'),
        MetricCallback(ce_loss_metric,metric_name='ce_loss'),
    ]
    if training_args.print_memory_stats:
        callbacks.append(MemoryCallback())
    trainer = SparseTrainer(
        model=model,
        args=training_args,
        train_dataset=iter_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        l2_act_metric=l2_act_metric,
        ce_loss_metric=ce_loss_metric,
        optimizers=optimizers,
    )
    trainer.train()
    trainer.save_model(training_args.model_output_path)

if __name__=="__main__":
    main()