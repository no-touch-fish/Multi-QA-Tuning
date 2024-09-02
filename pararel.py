import json
import random

input_file = 'dataset/pararel_train.json'
output_file = 'dataset/multiple_choice/pararel_train.json'

# read json file
with open(input_file, 'r') as file:
    data = json.load(file)

#  get question and answer
qa_pairs = []
for item in data:
    question = item[0]
    answer = item[1]
    if random.random() > 0.5:
        option = [f'{answer}','None of the above is true']
        answer = 'A'
    else:
        option = ['None of the above is true',f'{answer}']
        answer = 'B'
    qa_pairs.append({
        "question": question, 
        "options": option,
        "answer": answer
        })

# 输出整理后的question-answer形式
# for pair in qa_pairs:
#     print(f"Question: {pair['question']}\nAnswer: {pair['answer']}\n")

# save
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
print(f'the length of dataset is {len(qa_pairs)}')
print("数据已成功保存到本地JSON文件。")