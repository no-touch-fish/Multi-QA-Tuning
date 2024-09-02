from datasets import load_dataset
import json
input_file = 'dataset/mmlu_train.json'
output_file = 'dataset/multiple_choice/mmlu_train.json'

# read json file
with open(input_file, 'r') as file:
    data = json.load(file)

# get the question, choice and answer
my_dataset = []
for key in data:
    data_list = data[key]
    for item in data_list:
        question = item[0]
        choice = [item[1],item[2],item[3],item[4]]
        answer = item[5]
        my_dataset.append({
            'question': question,
            'options': choice,
            'answer':answer
        })


# 将数据写入本地JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(my_dataset, f, ensure_ascii=False, indent=4)

print(f'save to {output_file}')