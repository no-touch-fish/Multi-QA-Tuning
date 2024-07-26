import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import os

parser = argparse.ArgumentParser(
    description="""Sample unsuccessful test cases from the provided results 
    to create a new set of test cases with random data augmentation."""
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
    help = "which gpu to use",
)
parser.add_argument(
    "--save_path",
    type = str,
    help = "Path to the save dir",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
output_file = args.save_path

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

# load dataset
with open(data_file, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
# load model and tokenizer
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
llm = LLM(model=model_name)

# generate output
questions = df['question'].tolist()
answers = df['answer'].tolist()
sampling_params = SamplingParams(max_tokens=256)
results = llm.generate(questions, sampling_params)

# 处理生成的结果并保存
output_data = []
for question, result, answer in zip(questions, results, answers):
    generated_text = result.outputs[0].text
    output_data.append({'question': question, 'output': generated_text, 'answer': answer})

# save result
# output_file = 'result/llama3.json'
with open(output_file, 'w') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"结果已保存到 {output_file}")


