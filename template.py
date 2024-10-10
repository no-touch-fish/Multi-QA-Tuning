import json
import argparse

parser = argparse.ArgumentParser(
    description="""parameter"""
)
parser.add_argument(
    "--data_path",
    type = str,
    default = 'result/gsm.json',
    help = "Path to the dataset used.",
)
parser.add_argument(
    "--save_path",
    type = str,
    help = "Path to the save dir",
)
parser.add_argument(
    "--case",
    type = str,
    help = "multiple choice or blank",
)
parser.add_argument(
    "--question_number",
    type = int,
    default = 3,
    help = "the number for how many questions combine together",
)

args = parser.parse_args()

data_file = args.data_path
output_file = args.save_path
case = args.case
question_number = args.question_number

with open(data_file, 'r',encoding='utf-8') as file:
    data = json.load(file)

# deal with Question Answer setting
def blank(data):
    combined_data = []
    if question_number == 1:
        addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer.'
    elif question_number == 3:
        addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer \n2: answer \n3: answer.'
    elif question_number == 5:
        addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer \n2: answer \n3: answer \n4: answer \n5: answer.'
    # if the case is CoQA dataset
    count = 0
    if 'story' in data[0].keys():
        for subdata in data:
            # we only pick 5000 data from CoQA's training dataset
            if count > 5000:
                break
            for i in range(0, len(subdata['questions']), question_number):
                if i + question_number - 1 >= len(subdata['questions']):
                    break
                original_questions = []
                combined_question = f"{subdata['story']}\nSolve several independent questions here.\n"
                combined_answer = f''
                for j in range(0,question_number):
                    original_questions.append(subdata['questions'][i+j])
                    combined_question = combined_question + f'{j+1}: {subdata["questions"][i+j]} \n'
                    combined_answer = combined_answer + f'{subdata["answers"][i+j]} \n'
                combined_question = combined_question + f'{addtional_part}'
                count += 1
                combined_data.append({
                    "story": subdata['story'],
                    "original_questions": original_questions,
                    "question": combined_question, 
                    "answer": combined_answer
                    })
        return combined_data

    # apply templates to the original dataset
    for i in range(0, len(data), question_number):
        if i+question_number-1 >= len(data):
            break
        original_questions = []
        combined_question = f'Solve several independent questions here.\n'
        combined_answer = f''
        for j in range(0,question_number):
            original_questions.append(data[i+j]["question"])
            combined_question = combined_question + f'{j+1}: {data[i+j]["question"]}\n'
            combined_answer = combined_answer + f'{data[i+j]["answer"]} \n'
        combined_question = combined_question + f'{addtional_part}'
        combined_data.append({
            "original_questions": original_questions,
            "question": combined_question, 
            "answer": combined_answer
            })
    return combined_data

# deal with Multiple Choice dataset
def choice(data):
    combined_data = []
    if question_number == 1:
        addtional_part = 'Directly Give me an answer without explanation (which should be a letter in alphabet) for each question in following format: 1: answer.'
    elif question_number == 3:
        addtional_part = 'Directly Give me an answer without explanation (which should be a letter in alphabet) for each question in following format: 1: answer \n2: answer \n3: answer.'
    elif question_number == 5:
        addtional_part = 'Directly Give me an answer without explanation (which should be a letter in alphabet) for each question in following format: 1: answer \n2: answer \n3: answer \n4: answer \n5: answer.'
        # if the case is CoQA dataset
    if 'story' in data[0].keys():
        count = 0
        for subdata in data:
            if count > 5000:
                break
            for i in range(0, len(subdata['questions']), question_number):
                if i + question_number - 1 >= len(subdata['questions']):
                    break
                original_questions = []
                original_options = []
                combined_question = f"{subdata['story']}\nSolve several independent questions here.\n"
                combined_answer = f''
                for j in range(0,question_number):
                    original_questions.append(subdata['questions'][i+j])
                    original_options.append(subdata['options'][i+j])
                    combined_question = combined_question + f"{j+1}: {subdata['questions'][i+j]}\n" + f"options: {subdata['options'][i+j]}\n"
                    combined_answer = combined_answer + f"{subdata['answers'][i+j]} \n"
                combined_question = combined_question + f'{addtional_part}'
                count += 1
                combined_data.append({
                    "story": subdata['story'],
                    "original_questions": original_questions,
                    "original_options": original_options,
                    "question": combined_question, 
                    "answer": combined_answer
                    })
        return combined_data

    # apply templates to the original dataset
    for i in range(0, len(data), question_number):
        if i+question_number-1 >= len(data):
            break
        original_questions = []
        original_options = []
        combined_question = f'Solve several independent questions here.\n'
        combined_answer = f''
        for j in range(0,question_number):
            original_questions.append(data[i+j]["question"])
            original_options.append(data[i+j]["options"])
            combined_question = combined_question + f'{j+1}: {data[i+j]["question"]}\n' + f'options: {data[i+j]["options"]}\n'
            combined_answer = combined_answer + f'{data[i+j]["answer"]} \n'
        combined_question = combined_question + f'{addtional_part}'
        combined_data.append({
            "original_questions": original_questions,
            "original_options": original_options,
            "question": combined_question, 
            "answer": combined_answer
            })
    return combined_data

if case == 'blank':
    combined_data = blank(data)
else:
    combined_data = choice(data)

print('the number of this dataset is:',len(combined_data))

# Save the combined data to a new file
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print(f"save to file: {output_file}")

