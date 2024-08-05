from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import pandas as pd
from vllm import LLM, SamplingParams
import re

standard_score = 6

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
    help = "which gpu to use",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
# load the data
with open(data_file, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
questions = df['question'].tolist()
answers = df['answer'].tolist()
outputs = df['output'].tolist()
input_data = []
output_data = []
# load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_name)
# get the input
additional_part = "Remember what the correct answer is and use a score from 0 to 10 for each question to compare the generated output with the correct answer. \nThe format should be Score: 1: score 2: score 3: score. \n Score:"
for question,answer,output in zip(questions,answers,outputs):
    input = f"Question: {question} \nGenerated output: {output} \nCorrect answer: {answer} \n{additional_part}"
    input_data.append(input)
# get the score
sampling_params = SamplingParams(temperature = 0.0, max_tokens=100)
original_scores = llm.generate(input_data, sampling_params)
scores = []
for original_score in original_scores:
    score = original_score.outputs[0].text
    scores.append(score)
# get the label
labels = []
pattern = r'1: (\d+) 2: (\d+) 3: (\d+)'
for score in scores:
    label = []
    match = re.search(pattern, score)
    if match:
        for index in range(3):
            score_tmp = match.group(index + 1)
        if score_tmp >= standard_score:
            label.append(1) 
        else:
            label.append(0)
    else:
        print("wrong format for the score generation!!!")
    labels.append(label) 
# save to the output file
for question,answer,output,score,label in zip(questions,answers,outputs,scores,labels):
    output_data.append({
        'question': question, 
        'output': output, 
        'answer': answer,
        'score' : score,
        'label' : label
    })
#   print(response)
with open(data_file,'w') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"save to {data_file}")
