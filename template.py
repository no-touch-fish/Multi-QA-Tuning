import json
import argparse

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
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
output_file = args.save_path

template = 'Solve serveral independent questions here.'

with open(data_file, 'r',encoding='utf-8') as file:
    data = json.load(file)

# apply templates to every three lines of original dataset
addtional_part_1 = 'Give me one-word-answer (which should be a number) for each question in following format: 1: answer 2: answer 3: answer.'
combined_data = []
for i in range(0, len(data), 3):
    if i+2 >= len(data):
        break
    original_question = [data[i]["question"],data[i+1]["question"],data[i+2]["question"]]
    combined_question = f'{template} 1: {data[i]["question"]} \n2: {data[i+1]["question"]} \n3: {data[i+2]["question"]}\n{addtional_part_1}'
    combined_answer = f'{data[i]["answer"]} \n {data[i+1]["answer"]} \n {data[i+2]["answer"]}'
    combined_data.append({
        "original_questions": original_question,
        "question": combined_question, 
        "answer": combined_answer
        })

print('the number of this dataset is:',len(combined_data))

# Save the combined data to a new file
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print("数据已成功保存到本地JSON文件。")

