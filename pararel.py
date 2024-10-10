import json
import random

input_file = 'dataset/pararel_test.json'
output_file = 'dataset/blank/pararel_test.json'
random.seed(0)

# read json file
with open(input_file, 'r') as file:
    data = json.load(file)

#  get question and answer
qa_pairs = []
for item in data:
    question = item[0]
    answer = item[1]
    qa_pairs.append({
        "question": question, 
        "answer": answer
        })

# for pair in qa_pairs:
#     print(f"Question: {pair['question']}\nAnswer: {pair['answer']}\n")

# save to local file
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
print(f'the length of dataset is {len(qa_pairs)}')
print(f"save to {output_file}")