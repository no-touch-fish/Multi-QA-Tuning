from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import json
import argparse
import os
import random
from datasets import concatenate_datasets

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
    default = 2,
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
    default = 1,
    help = "the number for how many questions combine together",
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

template = 'Solve serveral independent questions here.'

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
    fp16=True,  # 启用FP16混合精度训练
    learning_rate= 1e-5,
    num_train_epochs= 3,
    logging_dir='/logs',
    save_total_limit=1,
)

# load the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = get_peft_model(model, lora_config)
# balance the certain and uncertain dataset
def balance_data(certain_data, uncertain_data):
    min_length = min(len(certain_data), len(uncertain_data))
    certain_data = certain_data[:min_length]
    uncertain_data = uncertain_data[:min_length]
    data = certain_data + uncertain_data
    return data

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
            combined_question = f'Solve serveral independent questions here.\n'
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
            combined_question = f'Solve serveral independent questions here.\n'
            combined_answer = f''
            combined_confidence = f''
            for j in range(0,question_number):
                combined_question = combined_question + f'{j+1}: {data[i+j]["question"]} \n' + f'options: {data[i+j]["options"]}\n'
                combined_answer = combined_answer + f'{j+1} {data[i+j]["answer"]} \n'
                combined_confidence = combined_confidence + f'{j+1}: {data[i+j]["confidence"]} \n'
            combined_question = combined_question + f'{addtional_part}'
            combined_data.append({
                "question": combined_question, 
                "answer": combined_answer,
                "confidence": combined_confidence
                })
    return combined_data

def tokenize_function_qa(example):
    prompt = 'Are you sure you accurately answered the question based on your internal knowledge?'
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\nQuestion:{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Answer:{example['answer']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # cut off
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
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
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\nQuestion:{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Answer:{example['answer']}.{prompt}{example['confidence']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # cut off
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
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
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\nQuestion:{example['question']}\nAnswer:{example['answer']}.{prompt}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['confidence']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # cut off
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
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
certain_file = f'{data_file}_certain.json'
uncertain_file = f'{data_file}_uncertain.json'
with open(certain_file, 'r',encoding='utf-8') as file:
    certain_data = json.load(file)
with open(uncertain_file, 'r',encoding='utf-8') as file:
    uncertain_data = json.load(file)
print(f'the length of certain:{len(certain_data)}, the length of uncertain:{len(uncertain_data)}')
# balance, combine and shuffle the dataset
data = certain_data + uncertain_data
# data = balance_data(certain_data,uncertain_data)
random.shuffle(data)
combine_data = preprocess_data(data)
# combine_data = data
data = Dataset.from_list(combine_data)
tokenized_data_confidence = data.map(tokenize_function_confidence)
tokenized_data_qa = data.map(tokenize_function_qa)

# combine two dataset together
# tokenized_data = concatenate_datasets([tokenized_data_qa, tokenized_data_confidence])
tokenized_data = data.map(tokenize_function_R)
print(f'the length of dataset is: {len(tokenized_data)}')

# fine tune
trainer = Trainer(
    model=model,  # model
    args=training_args,  # args
    train_dataset=tokenized_data,  # dataset
    # data_collator=data_collator, # data collator for padding
)
trainer.train()

# save the model
trainer.model.save_pretrained(output_file)
tokenizer.save_pretrained(output_file)

# test the data
save_data = []
for i, example in enumerate(tokenized_data):
        # 解码 input_ids 和 labels
        input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        filtered_labels = [label for label in example["labels"] if label >= 0]
        label_text = tokenizer.decode(filtered_labels, skip_special_tokens=True)
        save_data.append({
            'input': input_text,
            'label' : label_text
        })

output_file = 'dataset/test.json'
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
print('finish fine tune!')