import json
import argparse

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
    default = '0',
    help = "which gpu to use",
)
parser.add_argument(
    "--save_path",
    type = str,
    help = "Path to the save dir",
)
parser.add_argument(
    "--length",
    type = int,
    default = -1,
    help = "the length of the dataset",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
output_file = args.save_path
length = args.length


template = 'Solve serveral independent questions here.'

with open(data_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
if length == -1:
    length = len(data)
# apply templates to every three lines of original dataset
addtional_part = ' Directly give me one-word-answer for each question in following format: 1: choice 2: choice 3: choice. Your choice should be A,B,C,D,E,F,G,H,I,J,K...'
combined_data = []
for i in range(0, length, 3):
    if i+2 >= len(data):
        break
    original_questions = [data[i]["question"],data[i+1]["question"],data[i+2]["question"]]
    original_options = [data[i]["options"],data[i+1]["options"],data[i+2]["options"]]
    combined_question = f'{template} 1: {data[i]["question"]} \n options: {data[i]["options"]} \n 2: {data[i+1]["question"]} \n options: {data[i]["options"]} \n 3:{data[i+2]["question"]} \n options: {data[i]["options"]} \n' + addtional_part
    combined_answer = f'{data[i]["answer"]} \n {data[i+1]["answer"]} \n {data[i+2]["answer"]}'
    
    combined_data.append({
        "original_questions": original_questions,
        "original_options": original_options,
        "question": combined_question, 
        "answer": combined_answer
        })

print('the number of this dataset is:',len(combined_data))

# Save the combined data to a new file
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print(f'save to {output_file}')