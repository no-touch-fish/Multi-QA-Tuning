from datasets import load_dataset
import json
output_file = 'dataset/mmlu_test.json'
# 加载 MMLU 数据集
dataset = load_dataset("cais/mmlu",'all')
# length = 100
length = len(dataset['test'])
# 查看数据集的结构
# print(dataset)
# test_dataset = dataset['test'][:length]
answers_map = {'0':'A','1':'B','2':'C','3':'D'}
# 创建一个新的数据集列表
my_dataset = []

# 将前一百条数据添加到我的新数据集列表
for i in range(length):
    answer = dataset['test'][i]['answer']
    my_dataset.append({
        'question': dataset['test'][i]['question'],
        'options': dataset['test'][i]['choices'],
        'answer': answers_map[f'{answer}']
    })

# 将数据写入本地JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(my_dataset, f, ensure_ascii=False, indent=4)

print("the length of dataset is:",length)
print(f'save to {output_file}')