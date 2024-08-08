from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_checkpoint = "./results_packing/Final_Model_Checkpoint" # Use this one for your finetuned model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model.eval() # model in evaluation mode (dropout modules are deactivated)
print(model)

# # craft prompt (single prompt)
# comment = "What is the capital of France?" # Your prompt here.
# test_prompt=f'''<s>[INST] {comment} [/INST]'''

# Preparing Test Data List[str]

file_path = 'mistral/data/data_after_t_prompt_0.6sim.csv'

test_data = pd.read_csv(file_path, skiprows=range(1, 7393))

instructions = ["Translate the following sentence into a form suitable for a child: ",
                "Simplify the following so that a child will understand: ", 
                "Simplify the following: "]

for i, instruction in enumerate(instructions):

    prompts = ['[INST] ' + instruction + ad.strip("'\"") + ' [/INST]'for ad in test_data['adult_definition']]

    results = []
    for prompt in tqdm(prompts, desc='Generating', unit='Sentence'):
        # tokenize input
        inputs = tokenizer(prompt, return_tensors="pt") 

        # generate output
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs['attention_mask'].to('cuda'), max_new_tokens=140)

        #save output
        results.append(tokenizer.batch_decode(outputs)[0])

    output_df = pd.DataFrame(results)

    output_df.to_csv(f"mistral/data/inference_prompt_{i}.csv", header=False, index=False)

#print(tokenizer.batch_decode(outputs)[0])
