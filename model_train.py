import os
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
torch.manual_seed(42)



@dataclass
class ScriptArguments:

    local_rank: Optional[int] = -1 # change this to gpu id for server
    per_device_train_batch_size: Optional[int] = 4
    per_device_eval_batch_size: Optional[int] = 4
    gradient_accumulation_steps: Optional[int] = 4
    learning_rate: Optional[float] = 2e-5
    max_grad_norm: Optional[float] = 0.3
    weight_decay: Optional[int] = 0.01
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.1
    lora_r: Optional[int] = 32
    max_seq_length: Optional[int] = 512
    model_name: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct" # "meta-llama/Meta-Llama-3.1-8B-Instruct" # OR "mistralai/Mistral-7B-Instruct-v0.3" OR "tiiuae/falcon-7b-instruct"
    use_4bit: Optional[bool] = True
    use_nested_quant: Optional[bool] = False
    bnb_4bit_compute_dtype: Optional[str] = "bfloat16"
    bnb_4bit_quant_type: Optional[str] = "nf4"
    num_train_epochs: Optional[int] = 100
    fp16: Optional[bool] = False
    bf16: Optional[bool] = True
    packing: Optional[bool] = False
    gradient_checkpointing: Optional[bool] = True
    optim: Optional[str] = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    max_steps: int = 1000000
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    save_steps: int = 50
    logging_steps: int = 50
    merge_and_push: Optional[bool] = True # just saves the model's final checkpoint
    # output_dir: str = "./results_packing" # Mistral 
    # outpur_dir: str = "./falcon_results_packing"
    # output_dir: str = "./falcon_qwen_results_packing"
    # output_dir: str = "./falcon_gpt_results_packing"
    # output_dir: str = "./llama_results_packing"
    # output_dir: str = "./llama_qwen_results_packing"
    output_dir: str = "./llama_gpt_results_packing"

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



def gen_batches_train(val_ratio: float = 0.1):
    # file_path = 'data/data_after_t_prompt_0.6sim.csv'
    # file_path = 'data/from_Qwen2_final.csv'
    file_path = 'data/from_GPT_train_final.csv'

    # training_data = pd.read_csv(file_path, nrows=7392)
    # training_data = pd.read_csv(file_path, nrows=7718)
    training_data = pd.read_csv(file_path) # GPT

    instruction = 'Translate the following sentence into a form suitable for a child: '

    # prompts = ['<s>[INST] ' + instruction + ad.strip("'\"") + ' [/INST] ' + sim.strip("'\"") + ' </s>' for ad, sim in zip(training_data['adult_definition'], training_data['simplified'])]
    # prompts = [instruction + ad.strip("'\"") + sim.strip("'\"") for ad, sim in zip(training_data['adult_definition'], training_data['simplified'])] # Llama
    # prompts = ['<s>[INST] ' + instruction + ad.strip("'\"") + ' [/INST] ' + sim.strip("'\"") + ' </s>' for ad, sim in zip(training_data['adult_definition'], training_data['from_Qwen2'])]
    # prompts = [instruction + ad.strip("'\"") + sim.strip("'\"") for ad, sim in zip(training_data['adult_definition'], training_data['from_Qwen2'])] # Llama
    # prompts = ['<s>[INST] ' + instruction + ad.strip("'\"") + ' [/INST] ' + sim.strip("'\"") + ' </s>' for ad, sim in zip(training_data['adult_definition'], training_data['from_GPT'])]
    prompts = [instruction + ad.strip("'\"") + sim.strip("'\"") for ad, sim in zip(training_data['adult_definition'], training_data['from_GPT'])] # Llama

    total_samples = len(prompts)
    train_limit = int(total_samples * (1 - val_ratio))
    counter = 0

    for prompt in prompts:
        if counter >= train_limit:
            break
        yield {'text': prompt}
        counter += 1

def gen_batches_val(val_ratio: float = 0.1):
    # file_path = 'data/data_after_t_prompt_0.6sim.csv'
    # file_path = 'data/from_Qwen2_final.csv'
    file_path = 'data/from_GPT_train_final.csv'

    # training_data = pd.read_csv(file_path, nrows=7392)
    # training_data = pd.read_csv(file_path, nrows=7718)
    training_data = pd.read_csv(file_path) # GPT

    instruction = 'Translate the following sentence into a form suitable for a child: '

    # prompts = ['<s>[INST] ' + instruction + ad.strip("'\"") + ' [/INST] ' + sim.strip("'\"") + ' </s>' for ad, sim in zip(training_data['adult_definition'], training_data['simplified'])]
    # prompts = [instruction + ad.strip("'\"") + sim.strip("'\"") for ad, sim in zip(training_data['adult_definition'], training_data['simplified'])] # Llama
    # prompts = ['<s>[INST] ' + instruction + ad.strip("'\"") + ' [/INST] ' + sim.strip("'\"") + ' </s>' for ad, sim in zip(training_data['adult_definition'], training_data['from_Qwen2'])]
    # prompts = [instruction + ad.strip("'\"") + sim.strip("'\"") for ad, sim in zip(training_data['adult_definition'], training_data['from_Qwen2'])] # Llama
    # prompts = ['<s>[INST] ' + instruction + ad.strip("'\"") + ' [/INST] ' + sim.strip("'\"") + ' </s>' for ad, sim in zip(training_data['adult_definition'], training_data['from_GPT'])]
    prompts = [instruction + ad.strip("'\"") + sim.strip("'\"") for ad, sim in zip(training_data['adult_definition'], training_data['from_GPT'])] # Llama

    total_samples = len(prompts)
    train_limit = int(total_samples * (1 - val_ratio))
    counter = 0

    for prompt in prompts:
        if counter < train_limit:
            counter += 1
            continue
        if counter >= total_samples:
            break
        yield {'text': prompt}
        counter += 1


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        # use_auth_token=True,
        # revision="refs/pr/35"
    )

    model.config.pretraining_tp = 1
    model.config.window = 256

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # target_modules=["query_key_value"], # Use instead for Falcon
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer



# Training
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    evaluation_strategy="steps",
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False

train_gen = Dataset.from_generator(gen_batches_train)

val_gen = Dataset.from_generator(gen_batches_val)

print(train_gen)

print(val_gen)

print(model)

# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    eval_dataset=val_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()

# For merging checkpoint to main model, refer to model_merge.py ~