import json

input_file = 'dataset/pararel_original.json'
output_file = 'dataset/multiple_choice/pararel_test.json'

# 读取JSON文件
with open(input_file, 'r') as file:
    data = json.load(file)

# 提取question和answer
qa_pairs = []
for item in data:
    question = item[0]
    options = [item[1],'None of the above is true']
    # answer = item[1]
    answer = "A"
    qa_pairs.append({
        "question": question, 
        "options": options,
        "answer": answer
        })

# for pair in qa_pairs:
#     print(f"Question: {pair['question']}\nAnswer: {pair['answer']}\n")

# save
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

print("数据已成功保存到本地JSON文件。")