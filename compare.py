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
    "--case",
    type = str,
    default = None,
    help = "which case of result we are comparing",
)
parser.add_argument(
    "--question_number",
    type = int,
    default = 3,
    help = "the number for how many questions combine together",
)

args = parser.parse_args()
data_path = args.data_path
case = args.case
question_number = args.question_number

choice2num = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}
num2choice = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f'}

def blank(data):
    total = 0
    correct = 0
    for entry in data:
        answers = entry.get('answer', '').lower().split('\n')[:-1]
        outputs = entry.get('output', '').lower().split('\n')
        # check the answer
        labels = []
        for index,answer in enumerate(answers):
            right = 0
            for output in outputs:
                if (f'{index+1}: {answer.strip()}' in output):
                    right = 1
            if right == 1:
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

def choice(data):
    total = 0
    correct = 0
    for entry in data:
        answers = entry.get('answer', '').lower().split('\n')[:-1]
        outputs = entry.get('output', '').lower().split('\n')
        original_options = entry.get('original_options', '')
        # check the answer
        labels = []
        for index,answer in enumerate(answers):
            choice = choice2num[f'{answer.strip()}']
            original_answer = original_options[index][choice].lower()
            right = 0
            for output in outputs:
                if ((f'{index+1}:' in output or f'{index+1}.' in output or f'{num2choice[index]}:' in output) and (f'{answer.strip()}' in output or f'{original_answer}' in output)):
                    right = 1
            if right == 1:
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
if case == 'blank':
    updated_data = blank(data)
else:
    updated_data = choice(data)
# with the confidence
if 'confidence' in updated_data[0]:
    total = 0
    sure = 0
    unsure = 0
    correct = 0
    wrong = 0
    for entry in updated_data:
        label = entry.get('label', '')
        # find if "knowledge is there"
        i = entry['confidence'].lower().find('knowledge')
        if i == -1:
            confidences = entry['confidence'].lower().split('\n')[:question_number]
        else:
            confidence = entry['confidence']
            confidences = confidence[i:].lower().split('\n')[:question_number]
        if len(label) != len(confidences):
            total += question_number
            wrong += 3
            continue
        for index,confidence in enumerate(confidences):
            if (f'{index+1}:' in confidence and (f'unsure' in confidence or f'not sure' in confidence)):
                unsure += 1
                total += 1
            elif (f'{index+1}:' in confidence and f'sure' in confidence): # if you don't show unsure, then you are sure
                sure += 1
                total += 1
                correct += label[index]
            else:
                sure += 1
                total += 1
                correct += label[index]
    print(f'sure case:{sure}, unsure case:{unsure}, total case:{total}, Wrong case: {wrong}')
    print(f'successful rate:{100*correct/sure}%, answer rate: {100*sure/total}%')

# save the data
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Finish comparing")
