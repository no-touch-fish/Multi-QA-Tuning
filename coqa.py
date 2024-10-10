import json
import random

input_file = 'dataset/coqa_test.json'
output_file_1 = 'dataset/blank/coqa_test.json'
output_file_2 = 'dataset/multiple_choice/coqa_test.json'

random.seed(0)

# Question Answer setting
with open(input_file, 'r') as file:
    datas = json.load(file)

my_dataset = []
for data in datas['data']:
    my_dataset.append({
        'story': data["story"],
        'questions': [],
        'answers' : []
    })

with open(input_file, 'r') as file:
    datas = json.load(file)

for index,data in enumerate(datas['data']):
    for question, answer in zip(data['questions'],data['answers']):
        my_dataset[index]['questions'].append(question["input_text"])
        my_dataset[index]['answers'].append(answer["input_text"])

# write to json file
with open(output_file_1, 'w', encoding='utf-8') as f:
    json.dump(my_dataset, f, ensure_ascii=False, indent=4)

print(f'save to {output_file_1}')

# Multiple Choice setting
# with open(input_file, 'r') as file:
#     datas = json.load(file)

# my_dataset = []
# for data in datas['data']:
#     my_dataset.append({
#         'story': data["story"],
#         'questions': [],
#         'options':[],
#         'answers' : []
#     })

# with open(input_file, 'r') as file:
#     datas = json.load(file)

# for index,data in enumerate(datas['data']):
#     for question, answer in zip(data['questions'],data['answers']):
#         my_dataset[index]['questions'].append(question["input_text"])
#         random_number = random.random()
#         if random_number > 0.75:
#             option = [f'{answer["input_text"]}','None of the choices is true', 'All of the choices is true', 'Half of the choices is true']
#             answer = 'A'
#         elif random_number > 0.5 and random_number <= 0.75:
#             option = ['None of the choices is true', f'{answer["input_text"]}', 'All of the choices is true', 'Half of the choices is true']
#             answer = 'B'
#         elif random_number > 0.25 and random_number <= 0.5:
#             option = ['None of the choices is true', 'All of the choices is true', f'{answer["input_text"]}', 'Half of the choices is true']
#             answer = 'C'
#         else:
#             option = ['None of the choices is true', 'All of the choices is true', 'Half of the choices is true', f'{answer["input_text"]}']
#             answer = 'D'
#         my_dataset[index]['options'].append(option)
#         my_dataset[index]['answers'].append(answer)
        

# # write to json file
# with open(output_file_2, 'w', encoding='utf-8') as f:
#     json.dump(my_dataset, f, ensure_ascii=False, indent=4)

# print(f'save to {output_file_2}')