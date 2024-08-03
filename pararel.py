from datasets import load_dataset
import json
output_file = 'dataset/multiple_choice/pararel_test.json'
# 加载 MMLU 数据集
dataset = load_dataset("coastalcph/pararel_patterns")
# length = 3000
length = len(dataset['train'])
print("the length of dataset is:",length)
# 查看数据集的结构
# print(dataset)
# test_dataset = dataset['test'][:length]
answers_map = {'0':'A','1':'B','2':'C','3':'D','4':'E','5':'F'}
# 创建一个新的数据集列表
my_dataset = []

# 将前一百条数据添加到我的新数据集列表
for i in range(length):
    answer = dataset['train'][i]['object']
    for index in range(6):
        if dataset['train'][i]['candidates'][index] == answer:
            answer = answers_map[f'{index}']
            break
    my_dataset.append({
        'question': dataset['train'][i]['query'],
        'options': dataset['train'][i]['candidates'],
        'answer': answer
    })

# 将数据写入本地JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(my_dataset, f, ensure_ascii=False, indent=4)

print(f'save to {output_file}')