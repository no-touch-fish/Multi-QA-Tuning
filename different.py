import argparse
import json

parser = argparse.ArgumentParser(
    description="""Compare the result and the answer."""
)

parser.add_argument(
    "--data_path",
    type = str,
    help = "Path to the original result.",
)
parser.add_argument(
    "--fine_tune_data_path",
    type = str,
    help = "Path to the fine tune result.",
)

args = parser.parse_args()
data_path = args.data_path
fine_tune_data_path = args.fine_tune_data_path

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(fine_tune_data_path, 'r', encoding='utf-8') as f:
    fine_tune_data = json.load(f)

total = len(data) * len(data[0]['label'])
correct2wrong = 0
wrong2correct = 0
for result, fine_tune_result in zip(data, fine_tune_data):
    label = result['label']
    fine_tune_label = fine_tune_result['label']
    for i,j in zip(label, fine_tune_label):
        if i == 1 and j == 0:
            correct2wrong += 1
        if i == 0 and j == 1:
            wrong2correct += 1

print(f'correct2wrong:{correct2wrong}({correct2wrong/total:.4g}),wrong2correct:{wrong2correct}({wrong2correct/total:.4g}),total:{total}')