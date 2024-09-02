from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import argparse
import json
import os
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(
    description="""Compare the result and the answer."""
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
    default = '0',
    help = "which gpu to use",
)
parser.add_argument(
    "--case",
    type = str,
    default = None,
    help = "which case of result we are comparing",
)

args = parser.parse_args()
data_path = args.data_path
gpu = args.gpu
case = args.case
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

answer_map = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}

def keyword_extraction(data,case):
    pattern = r'1: (\d+) 2: (\d+) 3: (\d+)'
    total = 0
    correct = 0
    for entry in data:
        answers = entry.get('answer', '').lower().split('\n')
        output = entry.get('output', '').lower()
        original_options = entry.get('original_options', '')
        # check the answer
        labels = []
        for index,answer in enumerate(answers):
            choice = answer_map[f'{answer.strip()}']
            original_answer = original_options[index][choice].lower()
            if (f'{index+1}: {answer.strip()}' in output) or (f'{index+1}: {original_answer}' in output):
                labels.append(1)
                total += 1
                correct += 1
            else:
                labels.append(0)
                total += 1
        # add the label
        entry['label'] = labels
    print(f'total:{total},the original successful rate is:{100*correct/total}%')
    return data

# read the data
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# compare the answer
updated_data = keyword_extraction(data,case)
# with the confidence
if 'confidence' in updated_data[0]:
    total = 0
    sure = 0
    unsure = 0
    correct = 0
    for entry in updated_data:
        label = entry.get('label', '')
        confidence = entry['confidence']
        pattern = r"1:\s*([^2]*)2:\s*([^3]*)3:\s*([^1]*)"
        matches = re.findall(pattern, confidence)
        if matches:
            match = matches[-1]
            for index in range(3):
                confidence_tmp = match[index]
                if ('unsure' in confidence_tmp) or ('not sure' in confidence_tmp):
                    unsure += 1
                    total += 1
                else: # if you don't show unsure, then you are sure
                    sure += 1
                    total += 1
                    correct += label[index]
        else:
            print(f'String:{confidence} Wrong Format!')
    print(f'sure case:{sure}, unsure case:{unsure}, total case:{total}, successful rate:{100*correct/sure}%')

# save the data
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Finish comparing")