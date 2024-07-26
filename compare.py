import argparse
import json

parser = argparse.ArgumentParser(
    description="""Compare the result and the answer."""
)

parser.add_argument(
    "--data_path",
    type = str,
    default = 'dataset/processed_pararel_test.json',
    help = "Path to the dataset used.",
)
args = parser.parse_args()
data_path = args.data_path

def check_answers(data):
    total = 0
    correct = 0
    for entry in data:
        answers = entry.get('answer', '').lower().split('\n')
        output = entry.get('output', '').lower()
        # check the answer
        labels = []
        for answer in answers:
            if f'**{answer.strip()}**' in output:
                total += 1
                correct += 1
                labels.append(1)
            else:
                total += 1
                labels.append(0)
        # add the label
        entry['label'] = labels
    print(f'the successful rate is:{100*correct/total}%')
    return data

# read the data
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# compare the answer
updated_data = check_answers(data)

# save the data
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Finish comparing")