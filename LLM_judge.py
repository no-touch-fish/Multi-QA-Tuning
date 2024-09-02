from transformers import AutoTokenizer
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
parser.add_argument(
    "--save_path",
    type = str,
    default = None,
    help = "Path to the save dir",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 4,
    help = "the batch size for the input",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
output_file = args.save_path
if not output_file:
    output_file = data_file
batch_size = args.batch_size

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
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    max_num_seqs = batch_size
    )
# get the input
additional_part = "Use a score (which should be an integer) from 0 to 10 for each question to compare the generated output with the correct answer. 10 means matches and 0 means different.\nThe format should be Score: 1: score 2: score 3: score."
for question,answer,output in zip(questions,answers,outputs):
    input = [
        {
            "role" : "user",
            "content" : f"Question: {question} \nGenerated output: {output} \nCorrect answer: {answer} \n{additional_part}",
        }
    ]
    input_data.append(input)
inputs = tokenizer.apply_chat_template(
        conversation=input_data,
        add_generation_prompt=True,
    )
# get the score
sampling_params = SamplingParams(
    temperature = 0.0, 
    top_p = 1,
    max_tokens=100
    )
original_scores = llm.generate(
    prompt_token_ids=inputs, 
    sampling_params=sampling_params,
    use_tqdm=True,
    )
scores = []
for original_score in original_scores:
    score = original_score.outputs[0].text
    scores.append(score)

# get the label
labels = []
pattern = r'1: (\d+) 2: (\d+) 3: (\d+)'
total = 0
correct = 0
for score in scores:
    label = []
    match = re.search(pattern, score)
    if match:
        for index in range(3):
            score_tmp = match.group(index + 1).strip()
            if int(score_tmp) >= standard_score:
                total += 1
                correct += 1
                label.append(1) 
            else:
                total += 1
                label.append(0)
    else:
        print("wrong format for the score generation!!!")
        total += 3
        for index in range(3):
            label.append(0)
    labels.append(label) 
print(f'total:{total},original successful rate: {100*correct/total}%')
# get the confidence if it has
if 'confidence' in df.columns:
    pattern = r"1:\s*([^2]*)2:\s*([^3]*)3:\s*([^1]*)"
    confidences = df['confidence'].tolist()
    total = 0
    sure = 0
    unsure = 0
    correct = 0
    for confidence, label in zip(confidences,labels):
        matches = re.findall(pattern, confidence)
        if matches:
            match = matches[-1]
            for index in range(3):
                confidence_tmp = match[index]
                if ('unsure' in confidence_tmp) or ('not sure' in confidence_tmp):
                    total += 1
                    unsure += 1
                else: # if you don't show unsure, then you are sure
                    total += 1
                    sure += 1
                    correct += label[index]
        else:
            print(f'string:{confidence} Wrong Format!!!')
            for index in range(3):
                total += 1
                sure += 1
                correct += label[index]
    print(f'sure case:{sure}, unsure case:{unsure}, total case:{total}, successful rate:{100*correct/sure}%')
    
# save to the output file
for entry,score,label in zip(data,scores,labels):
    entry['score'] = score
    entry['label'] = label
#   print(response)
with open(output_file,'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"save to {output_file}")
