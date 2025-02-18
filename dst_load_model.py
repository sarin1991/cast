import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
config_path = "config.json"
CHECKPOINT_DIR = "/workspace/dual_decoder/mistral_out/checkpoint-10000/pytorch_model_fsdp_0"
output_dir = "/workspace/DualMistral_180M_1B"

config = AutoConfig.from_pretrained(config_path,attn_implementation="flash_attention_2")
model = AutoModelForCausalLM.from_config(config)

state_dict = {
    "model" : model.state_dict(),
}
DCP.load_state_dict(
    state_dict=state_dict,
    storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
    no_dist=True,
)
model.load_state_dict(state_dict["model"])
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_dir)