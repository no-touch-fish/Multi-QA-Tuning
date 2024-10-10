import json
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(
    description="""parameter"""
)
parser.add_argument(
    "--data_path",
    type = str,
    default = 'result/gsm.json',
    help = "Path to the dataset used.",
)
parser.add_argument(
    "--save_path",
    type = str,
    help = "Path to the save dir",
)
parser.add_argument(
    "--result_path",
    type = str,
    help = "Path to the result file",
)
parser.add_argument(
    "--case",
    type = str,
    default = 'choice',
    help = "choice or blank",
)
args = parser.parse_args()

data_file = args.data_path
output_certain_file = f'{args.save_path}_certain.json'
output_uncertain_file = f'{args.save_path}_uncertain.json'
# output_train_file = f'{args.save_path}_train.json'
result_file = args.result_path
case = args.case

# deal with CoQA dataset
def divide_CoQA(storys):
    certain_data = []
    uncertain_data = []
    if case == 'choice':
        choices = df['original_options'].tolist()
        for label, question, choice, answer, story in zip(labels, questions, choices, answers, storys):
            subdata = {
                "story": story,
                "questions": [], 
                "options": [],
                "answers": [],
                "confidences": []
            }
            answer = answer.split('\n')
            for sublabel, subquestion, subchoice, subanswer in zip(label, question, choice, answer):
                if sublabel == 1:
                    subanswer = f'{subanswer.strip()}'
                    subdata['questions'].append(subquestion)
                    subdata['options'].append(subchoice)
                    subdata['answers'].append(subanswer)
                    subdata['confidences'].append('I am sure')
                else:
                    subanswer = f'{subanswer.strip()}'
                    subdata['questions'].append(subquestion)
                    subdata['options'].append(subchoice)
                    subdata['answers'].append(subanswer)
                    subdata['confidences'].append('I am unsure')
            certain_data.append(subdata)
    elif case == 'blank':
        for label, question, answer, story in zip(labels, questions, answers, storys):
            subdata = {
                "story": story,
                "questions": [], 
                "answers": [],
                "confidences": []
            }
            answer = answer.split('\n')
            for sublabel, subquestion, subanswer in zip(label, question, answer):
                if sublabel == 1:
                    subanswer = f'{subanswer.strip()}'
                    subdata['questions'].append(subquestion)
                    subdata['answers'].append(subanswer)
                    subdata['confidences'].append('I am sure')
                else:
                    subanswer = f'{subanswer.strip()}'
                    subanswer = f'{subanswer.strip()}'
                    subdata['questions'].append(subquestion)
                    subdata['answers'].append(subanswer)
                    subdata['confidences'].append('I am unsure')
            certain_data.append(subdata)
    return certain_data, uncertain_data

# get the label out
with open(result_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame(data)
labels = df['label'].tolist()
# get the question and answer
with open(data_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
df = pd.DataFrame(data)
questions = df['original_questions'].tolist()
answers = df['answer'].tolist()
# divide the data
certain_data = []
uncertain_data = []
if case == 'choice':
    choices = df['original_options'].tolist()
    for label, question, choice, answer in zip(labels, questions, choices, answers):
        answer = answer.split('\n')
        for sublabel, subquestion, subchoice, subanswer in zip(label, question, choice, answer):
            if sublabel == 1:
                subanswer = f'{subanswer.strip()}'
                certain_data.append({
                    "question": subquestion, 
                    "options": subchoice,
                    "answer": subanswer,
                    "confidence": 'I am sure'
                })
            else:
                subanswer = f'{subanswer.strip()}'
                uncertain_data.append({
                    "question": subquestion, 
                    "options": subchoice,
                    "answer": subanswer,
                    "confidence": 'I am unsure'
                })
elif case == 'blank':
    for label, question, answer in zip(labels, questions, answers):
        answer = answer.split('\n')
        for sublabel, subquestion, subanswer in zip(label, question, answer):
            if sublabel == 1:
                subanswer = f'{subanswer.strip()}'
                certain_data.append({
                    "question": subquestion, 
                    "answer": subanswer,
                    "confidence": 'I am sure'
                })
            else:
                subanswer = f'{subanswer.strip()}'
                uncertain_data.append({
                    "question": subquestion, 
                    "answer": subanswer,
                    "confidence": 'I am unsure'
                })

if 'story' in df.keys():
    storys = df['story'].tolist()
    # print(storys)
    certain_data, uncertain_data = divide_CoQA(storys)

print('the length of certain data is:',len(certain_data))
print('the length of uncertain data is:',len(uncertain_data))
# save the file
os.makedirs(os.path.dirname(output_certain_file), exist_ok=True)
os.makedirs(os.path.dirname(output_uncertain_file), exist_ok=True)
# os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
with open(output_certain_file, 'w',encoding='utf-8') as f:
    json.dump(certain_data, f, indent=4, ensure_ascii=False)
with open(output_uncertain_file, 'w',encoding='utf-8') as f:
    json.dump(uncertain_data, f, indent=4, ensure_ascii=False)

print("save to local file.")


