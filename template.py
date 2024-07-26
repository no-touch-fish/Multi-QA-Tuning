import json
import argparse

input_file = 'dataset/pararel_test.json'
output_file = 'dataset/processed_pararel_test.json'

template = 'Solve serveral independent questions here.'

with open(input_file, 'r',encoding='utf-8') as file:
    data = json.load(file)

# apply templates to every three lines of original dataset
addtional_part = ' Answer these questions. Please mark the answer with ** so I can quickly find it.'
combined_data = []
for i in range(0, len(data), 3):
    if i+2 >= len(data):
        break
    combined_question = f'{template} 1: {data[i]["question"]} \n 2: {data[i+1]["question"]} \n 3:{data[i+2]["question"]}' + addtional_part
    combined_answer = f'{data[i]["answer"]} \n {data[i+1]["answer"]} \n {data[i+2]["answer"]}'
    combined_data.append({"question": combined_question, "answer": combined_answer})

print('the number of this dataset is:',len(combined_data))

# Save the combined data to a new file
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print("数据已成功保存到本地JSON文件。")

