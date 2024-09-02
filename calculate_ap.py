import pandas as pd
import torch
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
import os
import numpy as np
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(
    description="""parameter"""
)
parser.add_argument(
    "--data_path",
    type = str,
    default = None,
    help = "Path to the original dataset used.",
)
parser.add_argument(
    "--gpu",
    type = str,
    help = "which gpu to use",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 4,
    help = "the batch size for the input",
)
parser.add_argument(
    "--case",
    type = str,
    default = 'choice',
    help = "choice or blank",
)
parser.add_argument(
    '--lora_path',
    type = str,
    default = None,
    help = "the path to the lora model",
)

args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
case = args.case
batch_size = args.batch_size
lora_path = args.lora_path

with open(data_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame(data)
questions = df['question'].tolist()
outputs = df['output'].tolist()
labels = df['label'].tolist()

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        logprobs=5,
    )
model = LLM(
        model=model_name, 
        enable_lora=True,
        max_num_seqs = 32,
    )

# function to divide the answer from the output
def divide_content(content):
    output_list = []
    index_1 = content.find('1:')
    index_2 = content.find('2:')
    index_3 = content.find('3:')
    index_end = content[index_3:].find('Directly')
    if index_1 == -1 or index_2 == -1 or index_3 == -1:
        return None
    output_list.append(content[index_1+3:index_2-1])
    output_list.append(content[index_2+3:index_3-1])
    output_list.append(content[index_3+3:index_end-1])
    return output_list

# function to get the prob of first index
def get_prob(logprobs):
    prob_list = []
    for index,logprob in enumerate(logprobs):
        if index > 1:
            break
        for key in logprobs[0]:
            if logprob[key].decoded_token == ' sure':
                prob = math.exp(logprob[key].logprob)
                prob_list.append({' sure':prob})
            elif logprob[key].decoded_token == ' unsure':
                prob = math.exp(logprob[key].logprob)
                prob_list.append({' unsure':prob})
    key_list = []
    for prob in prob_list:
        key_list.extend(list(prob.keys()))
    if ' sure' not in key_list:
        prob_list.append({' sure': 0})
    if ' unsure' not in key_list:
        prob_list.append({' unsure': 0})
    return prob_list

# function to get AP score
def calculate_scores(label,prob):
        p1, r1, _ = precision_recall_curve(label, prob)
        ap_score = average_precision_score(label, prob)
        return ap_score

# get the prompt for Q-A pair
prompt = []
original_questions = []
additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? I am'
for question, output in zip(questions, outputs):
    question_list = divide_content(question)
    output_list = divide_content(output)
    if output_list == None:
        print('wrong answer format, not count!')
        continue
    for index in range(3):
        prompt.append(f'Question:{question_list[index]},Answer:{output_list[index]}.{additional_part}')
        original_questions.append(question_list[index])

results = model.generate(
        prompt,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("lora_adapter", 1, lora_path)
        )
confidences = []
probs = []
for result in results:
        confidences.append(result.outputs[0].text)
        probs.append(get_prob(result.outputs[0].logprobs))
    
output_data = []
for index, (question,confidence,prob) in enumerate(zip(original_questions,confidences,probs)):
    output_data.append({
        'question':question,
        'confidence':confidence,
        'prob':prob
    })

# save the result
output_file = 'fine_tune_result/test.json'
with open(output_file, 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"save the result to {output_file}")

# calculate the AP score
input_label = []
input_prob = []
for label in labels:
    input_label.extend(label)
for prob in probs:
    input_label.append(0.5*(prob[' sure'] + prob[' unsure']))

score = calculate_scores(input_label,input_prob)
print(f"Average Precision Score: {score}")

# save the file to check the prompt
# output_file = 'fine_tune_result/test.json'
# with open(output_file, 'w') as file:
#     json.dump(prompt, file, ensure_ascii=False, indent=4)

# print(f"save the result to {output_file}")








    



