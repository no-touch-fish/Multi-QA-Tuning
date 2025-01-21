from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import json
import argparse
import os
import random
from datasets import concatenate_datasets,interleave_datasets

random.seed(0)

parser = argparse.ArgumentParser(
    description="""parameter"""
)

parser.add_argument(
    "--data_path",
    type = str,
    default = 'dataset/processed_pararel_test.json',
    help = "Path to the dataset used.",
)
parser.add_argument(
    "--gpu",
    type = str,
    default = None,
    help = "which gpu to use",
)
parser.add_argument(
    "--save_path",
    type = str,
    help = "Path to the save directory",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 1,
    help = "the batch size for the input",
)
parser.add_argument(
    "--case",
    type = str,
    default = 'choice',
    help = "choice or blank",
)
parser.add_argument(
    "--question_number",
    type = int,
    default = 3,
    help = "the number for how many questions combine together",
)
parser.add_argument(
    '--MTI', 
    action='store_true', 
    help="too lazy to add a new dataset",
)

args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
output_file = args.save_path
os.makedirs(os.path.dirname(output_file), exist_ok=True)
batch_size = args.batch_size
case = args.case
question_number = args.question_number

template = 'Solve several questions here.'
MAX_LENGTH = 1200
batch_size = 1

# Lora config
lora_config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM,
    r=8,               # rank
    lora_alpha=32,     # alpha
    inference_mode=False,
    target_modules=["q_proj", "v_proj"],  # module to add lora
    lora_dropout=0.1   # dropout
)
# training config
training_args = TrainingArguments(
    output_dir= output_file,
    per_device_train_batch_size= batch_size, # batch size
    gradient_accumulation_steps=4,
    fp16=True,  # FP16 mix
    learning_rate= 1e-5,
    num_train_epochs= 3,
    logging_dir='/logs',
    save_total_limit=1,
)

# load the model and the tokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = 'Qwen/Qwen2-7B-Instruct'

model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

model = get_peft_model(model, lora_config)

# deal with MTI dataset
def preprocess_MTI(data_file):
    combine_data = []
    with open(data_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    print(f'the length of data is:{len(data)}')
    for subdata in data:
        answers = subdata['answer'].split('\n')
        confidence = []
        for i in range(3):
            if subdata['label'][i] == 1:
                confidence.append('I am sure')
            else:
                confidence.append('I am unsure')
        combine_data.append({
            "question": subdata['question'], 
            "answer": f'1: {answers[0]} \n2: {answers[1]} \n3: {answers[2]}',
            "confidence": f'1: {confidence[0]} \n2: {confidence[1]} \n3: {confidence[2]}'
        })
    return combine_data

# combine data from CoQA together
def preprocess_CoQA(data):
    combined_data = []
    if case == 'blank':
        if question_number == 1:
            addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer.'
        elif question_number == 3:
            addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer \n2: answer \n3: answer.'
        elif question_number == 5:
            addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer \n2: answer \n3: answer \n4: answer \n5: answer.'
        # apply templates to the original dataset
        for subdata in data:
            for i in range(0, len(subdata['questions']), question_number):
                if i+question_number-1 >= len(data):
                    break
                combined_question = f"{subdata['story']}\n" + f'Solve serveral questions here.\n'
                combined_answer = f''
                combined_confidence = f''
                for j in range(0,question_number):
                    combined_question = combined_question + f'{j+1}: {subdata["questions"][i+j]} \n'
                    combined_answer = combined_answer + f'{j+1}: {subdata["answers"][i+j]} \n'
                    combined_confidence = combined_confidence + f'{j+1}: {subdata["confidences"][i+j]} \n'
                combined_question = combined_question + f'{addtional_part}'
                combined_data.append({
                    "question": combined_question, 
                    "answer": combined_answer,
                    "confidence": combined_confidence
                })
    elif case == 'choice':
        if question_number == 1:
            addtional_part = 'Directly Give me a choice (which should be a letter from the alphabet) for each question in following format: 1: choice.'
        elif question_number == 3:
            addtional_part = 'Directly Give me a choice (which should be a letter from the alphabet) for each question in following format: 1: choice \n2: choice \n3: choice.'
        elif question_number == 5:
            addtional_part = 'Directly Give me a choice (which should be a letter from the alphabet) for each question in following format: 1: choice \n2: choice \n3: choice \n4: choice \n5: choice.'
        # apply templates to the original dataset
        for subdata in data:
            for i in range(0, len(subdata['questions']), question_number):
                if i+question_number-1 >= len(data):
                    break
                combined_question = f"{subdata['story']}\n" + f'Solve several questions here.\n'
                combined_answer = f''
                combined_confidence = f''
                for j in range(0,question_number):
                    combined_question = combined_question + f'{j+1}: {subdata["questions"][i+j]} \n' + f"options: {subdata['options'][i+j]}\n"
                    combined_answer = combined_answer + f'{j+1}: {subdata["answers"][i+j]} \n'
                    combined_confidence = combined_confidence + f'{j+1}: {subdata["confidences"][i+j]} \n'
                combined_question = combined_question + f'{addtional_part}'
                combined_data.append({
                    "question": combined_question, 
                    "answer": combined_answer,
                    "confidence": combined_confidence
                })
    return combined_data

# combine data from CoT format together
def preprocess_CoT(data):
    combined_data = []
    if case == 'blank':
        if question_number == 1:
            addtional_part = "Let's think step by step and give me an answer for each question in following format: 1: answer."
        elif question_number == 3:
            addtional_part = "Let's think step by step and give me an answer for each question in following format: 1: answer \n2: answer \n3: answer."
        elif question_number == 5:
            addtional_part = "Let's think step by step and give me an answer for each question in following format: 1: answer \n2: answer \n3: answer \n4: answer \n5: answer."
        # apply templates to the original dataset
        for i in range(0, len(data), question_number):
            if i+question_number-1 >= len(data):
                break
            combined_question = f'{data[0]["context"]}Solve serveral questions here.\n'
            # combined_question = f'Solve serveral questions here.\n'
            combined_answer = f''
            combined_confidence = f''
            for j in range(0,question_number):
                combined_question = combined_question + f'{j+1}: {data[i+j]["question"]} \n'
                combined_answer = combined_answer + f'{j+1}: {data[i+j]["cot"]} \n'
                combined_confidence = combined_confidence + f'{j+1}: {data[i+j]["confidence"]} \n'
            combined_question = combined_question + f'{addtional_part}'
            combined_data.append({
                "question": combined_question, 
                "answer": combined_answer,
                "confidence": combined_confidence
            })
    return combined_data

# combine the data together
def preprocess_data(data):
    combined_data = []
    if case == 'blank':
        if question_number == 1:
            addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer.'
        elif question_number == 3:
            addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer \n2: answer \n3: answer.'
        elif question_number == 5:
            addtional_part = 'Directly Give me an answer without explanation for each question in following format: 1: answer \n2: answer \n3: answer \n4: answer \n5: answer.'
        # apply templates to the original dataset
        for i in range(0, len(data), question_number):
            if i+question_number-1 >= len(data):
                break
            combined_question = f'Solve serveral questions here.\n'
            combined_answer = f''
            combined_confidence = f''
            for j in range(0,question_number):
                combined_question = combined_question + f'{j+1}: {data[i+j]["question"]} \n'
                combined_answer = combined_answer + f'{j+1}: {data[i+j]["answer"]} \n'
                combined_confidence = combined_confidence + f'{j+1}: {data[i+j]["confidence"]} \n'
            combined_question = combined_question + f'{addtional_part}'
            combined_data.append({
                "question": combined_question, 
                "answer": combined_answer,
                "confidence": combined_confidence
            })
    elif case == 'choice':
        if question_number == 1:
            addtional_part = 'Directly Give me a choice (which should be a letter from the alphabet) for each question in following format: 1: choice.'
        elif question_number == 3:
            addtional_part = 'Directly Give me a choice (which should be a letter from the alphabet) for each question in following format: 1: choice \n2: choice \n3: choice.'
        elif question_number == 5:
            addtional_part = 'Directly Give me a choice (which should be a letter from the alphabet) for each question in following format: 1: choice \n2: choice \n3: choice \n4: choice \n5: choice.'
        # apply templates to the original dataset
        for i in range(0, len(data), question_number):
            if i+question_number-1 >= len(data):
                break
            combined_question = f'Solve several questions here.\n'
            combined_answer = f''
            combined_confidence = f''
            for j in range(0,question_number):
                combined_question = combined_question + f'{j+1}: {data[i+j]["question"]} \n' + f'options: {data[i+j]["options"]}\n'
                combined_answer = combined_answer + f'{j+1}: {data[i+j]["answer"]} \n'
                combined_confidence = combined_confidence + f'{j+1}: {data[i+j]["confidence"]} \n'
            combined_question = combined_question + f'{addtional_part}'
            combined_data.append({
                "question": combined_question, 
                "answer": combined_answer,
                "confidence": combined_confidence
                })
    return combined_data

def tokenize_function_qa(example):
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|im_start|>user\nQuestion:{example['question']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Answer:{example['answer']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # cut off
        print('truncation happen')
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        # input_ids = input_ids[:MAX_LENGTH]
        # attention_mask = attention_mask[:MAX_LENGTH]
        # labels = labels[:MAX_LENGTH]
    elif len(input_ids) < MAX_LENGTH:  # padding
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def tokenize_function_R(example):
    prompt = 'Are you sure you accurately answered the question based on your internal knowledge?'
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|im_start|>user\nQuestion:{example['question']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # don't add special tokens at the beginning
    response = tokenizer(f"Answer:{example['answer']}.{prompt}{example['confidence']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # eos token!
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # cut off
        print('truncation happen')
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        # input_ids = input_ids[:MAX_LENGTH]
        # attention_mask = attention_mask[:MAX_LENGTH]
        # labels = labels[:MAX_LENGTH]
    elif len(input_ids) < MAX_LENGTH:  # padding
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def tokenize_function_confidence(example):
    prompt = 'Are you sure you accurately answered the question based on your internal knowledge?'
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|im_start|>user\nQuestion:{example['question']}\nAnswer:{example['answer']}.{prompt}.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # don't add special tokens at the beginning
    response = tokenizer(f"{example['confidence']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # eos token!
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # cut off
        print('truncation happen')
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        # input_ids = input_ids[:MAX_LENGTH]
        # attention_mask = attention_mask[:MAX_LENGTH]
        # labels = labels[:MAX_LENGTH]
    elif len(input_ids) < MAX_LENGTH:  # padding
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# load the data
if args.MTI:
    data = preprocess_MTI(data_file)
else:
    certain_file = f'{data_file}_certain.json'
    uncertain_file = f'{data_file}_uncertain.json'
    with open(certain_file, 'r',encoding='utf-8') as file:
        certain_data = json.load(file)
    with open(uncertain_file, 'r',encoding='utf-8') as file:
        uncertain_data = json.load(file)
    print(f'the length of certain:{len(certain_data)}, the length of uncertain:{len(uncertain_data)}')
    # get the multiple problem dataset
    data = certain_data + uncertain_data
    random.shuffle(data)

if args.MTI:
    combine_data = data
elif 'story' in data[0].keys():
    combine_data = preprocess_CoQA(data)
elif 'cot' in data[0].keys():
    combine_data = preprocess_CoT(data)
else:
   combine_data = preprocess_data(data)

data = Dataset.from_list(combine_data)
tokenized_data_confidence = data.map(tokenize_function_confidence)
tokenized_data_qa = data.map(tokenize_function_qa)

# MP Tuning
print('Doing MP Tuning!')
tokenized_data = concatenate_datasets([tokenized_data_qa, tokenized_data_confidence])
# tokenized_data = interleave_datasets([tokenized_data_qa, tokenized_data_confidence])
tokenized_data = tokenized_data.filter(lambda x: len(x["input_ids"]) > 0)

# R-Tuning (S: question number = 1 / M: question number = 3)
# print(f'Doing R-Tuning with number: {question_number}')
# tokenized_data = data.map(tokenize_function_R)
# tokenized_data = tokenized_data.filter(lambda x: len(x["input_ids"]) > 0)

# Vanilla
# print(f'Doing Vanilla')
# tokenized_data = tokenized_data_qa
# tokenized_data = tokenized_data.filter(lambda x: len(x["input_ids"]) > 0)

print(f'the length of dataset is: {len(tokenized_data)}')

# check the training data
save_data = []
for i, example in enumerate(tokenized_data):
        # decode input_ids and labels
        input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        filtered_labels = [label for label in example["labels"] if label >= 0]
        label_text = tokenizer.decode(filtered_labels, skip_special_tokens=True)
        save_data.append({
            'input': input_text,
            'label' : label_text
        })

data_output_file = 'dataset/test.json'
with open(data_output_file, 'w',encoding='utf-8') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)

# fine tune
trainer = Trainer(
    model=model,  # model
    args=training_args,  # args
    train_dataset=tokenized_data,  # dataset
)
trainer.train()

# save the model
trainer.model.save_pretrained(output_file)
tokenizer.save_pretrained(output_file)

print('finish fine tune!')