from datasets import load_dataset

# 加载 MMLU 数据集
dataset = load_dataset("cais/mmlu",)

# 现在你可以访问数据集的不同部分，例如训练集、验证集和测试集
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# 打印数据集的结构
print(dataset)

# 打印测试集中前5个样本的问题、选项和答案
for i in range(5):
    print(f"问题 {i+1}: {test_dataset[i]['question']}")
    print(f"选项: {test_dataset[i]['options']}")
    print(f"正确答案: {test_dataset[i]['answer']}\n")