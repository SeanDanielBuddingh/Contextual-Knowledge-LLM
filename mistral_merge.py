import os
from dataclasses import dataclass
import torch
from peft import AutoPeftModelForCausalLM
from transformers import HfArgumentParser

torch.manual_seed(42)

@dataclass
class ScriptArguments:
    output_dir: str = "./results_packing"

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

output_dir = os.path.join(script_args.output_dir, "checkpoint-2000") #change this to your selected checkpoint based on evaluation -- check wandb, use wadnb.ipynb 

torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.output_dir, "Final_Model_Checkpoint") #change name to avoid overwriting
model.save_pretrained(output_merged_dir, safe_serialization=True)