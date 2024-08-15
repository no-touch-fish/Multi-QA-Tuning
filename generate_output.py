import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
import os

parser = argparse.ArgumentParser(
    description="""parameter"""
)

parser.add_argument(
    "--data_path",
    type = str,
    default = 'dataset/processed_pararel_test.json',
    help = "Path to the dataset used.",
)
parser.add_argument(
    "--gpu",
    type = str,
    default = None,
    help = "which gpu to use",
)
parser.add_argument(
    "--save_path",
    type = str,
    help = "Path to the save dir",
)
parser.add_argument(
    "--case",
    type = str,
    default = 'choice',
    help = "choice or blank",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 4,
    help = "the batch size for the input",
)
parser.add_argument(
    '--generate_vllm', 
    action='store_true', 
    help="Generate using vllm",
)
parser.add_argument(
    '--lora_model', 
    action='store_true', 
    help="Generate using lora fine-tuned model",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = args.save_path
case = args.case
batch_size = args.batch_size

# load dataset
with open(data_file, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
questions = df['question'].tolist()
answers = df['answer'].tolist()
# load model and tokenizer
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
def get_generate_input(questions,model_name):
    inputs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
            [
                {
                    "role": "user",
                    "content": question,
                }
            ] for question in questions
        ]

    inputs = tokenizer.apply_chat_template(
        conversation=prompts,
        add_generation_prompt=True,
    )
    return inputs

def generate_vllm(inputs,model_name,batch_size):
    llm = LLM(
        model= model_name,
        max_num_seqs = batch_size
        )
    sampling_params = SamplingParams(
        temperature = 0.0, 
        top_p = 1,
        max_tokens=64,
        )
    results = llm.generate(
        prompt_token_ids = inputs,
        sampling_params = sampling_params,
        use_tqdm = True,
        )
    generation = []
    for result in results:
        generation.append(result.outputs[0].text)
    return generation

def generate_lora(inputs,model_name,batch_size):
    lora_file = "models/llama3_gsm"
    generation = []
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        )
    model = LLM(
        model=model_name, 
        enable_lora=True,
        max_num_seqs = batch_size
        )
    results = model.generate(
        prompt_token_ids=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("lora_adapter", 1, lora_file)
        )
    for result in results:
        generation.append(result.outputs[0].text)
    return generation

# generate the output
inputs = get_generate_input(questions,model_name)
if args.generate_vllm:
    generations = generate_vllm(inputs,model_name,batch_size)
else:
    generations = generate_lora(inputs,model_name,batch_size)

# save the output
output_data = []
if case == 'choice':
    original_options = df['original_options'].tolist()
    for question, generation, answer,original_option in zip(questions, generations, answers,original_options):
        output_data.append({
            'question': question, 
            'original_options':original_option,
            'output': generation, 
            'answer': answer
        })
elif case == 'blank':
    for question, generation, answer in zip(questions, generations, answers):
        output_data.append({
            'question': question, 
            'output': generation, 
            'answer': answer
            })
with open(output_file, 'w') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"save the result to {output_file}")


