from datasets import load_dataset
import json

output_file = 'dataset/mmlu_pro_test.json'
# 加载数据集
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
length = len(dataset['test'])
# 查看数据集的结构
# print(dataset)
# test_dataset = dataset['test'][:length]

# 创建一个新的数据集列表
my_dataset = []

# 将前一百条数据添加到我的新数据集列表
for i in range(length):
    # print(i)
    my_dataset.append({
        'question': dataset['test'][i]['question'],
        'options': dataset['test'][i]['choices'],
        'answer': dataset['test'][i]['answer']
    })

# 将数据写入本地JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(my_dataset, f, ensure_ascii=False, indent=4)

print("the length of dataset is:",length)
print(f'save to {output_file}')

