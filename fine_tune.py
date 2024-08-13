from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch
import json
import argparse
import os
from fastchat.model import get_conversation_template

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
    default = 32,
    help = "the batch size for the input",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
output_file = args.save_path
os.makedirs(os.path.dirname(output_file), exist_ok=True)
batch_size = args.batch_size

template = 'Solve serveral independent questions here.'

# Lora config
lora_config = LoraConfig(
    r=8,               # rank
    lora_alpha=32,     # alpha
    target_modules=["q_proj", "v_proj"],  # module to add lora
    lora_dropout=0.1   # dropout
)
# training config
training_args = TrainingArguments(
    output_dir= output_file,
    per_device_train_batch_size= batch_size,
    learning_rate= 2e-5,
    num_train_epochs= 3,
    logging_dir='/logs',
    save_total_limit=1,
)

# load the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = get_peft_model(model, lora_config)

def preprocess_data(data):
# apply templates to every three lines of original dataset
    combined_data = []
    
    for i in range(0, len(data), 3):
        if i+2 >= len(data):
            break
        conversation = get_conversation_template(model_name)
        combined_question = f'{template} 1: {data[i]["question"]} \n 2: {data[i+1]["question"]} \n 3:{data[i+2]["question"]}\n'
        combined_answer = f'{data[i]["answer"]} \n {data[i+1]["answer"]} \n {data[i+2]["answer"]}'
        conversation.append_message(
            conversation.roles[0],
            combined_question,
            )
        conversation.append_message(
            conversation.roles[1],
            None,
            )
        combined_data.append({
            "question": conversation.get_prompt(), 
            "answer": combined_answer
        })
    return combined_data

def tokenize_function(data):
    inputs = []
    questions = []
    answers = []
    for i in range(len(data)):
        questions.append(data[i]["question"])
        answers.append(data[i]["answer"])
    prompts = [
            [
                {
                    "role": "user",
                    "content": question,
                }
            ] for question in questions
        ]
    question_texts = tokenizer.apply_chat_template(
        conversation=prompts,
        add_generation_prompt=True,
        padding = True,
        tokenize=False,
    )
    questions = tokenizer(
        question_texts,
        padding="max_length",
        max_length = 256
    )
    prompts = [
            [
                {
                    "role": "assisstant",
                    "content": answer,
                }
            ] for answer in answers
        ]
    labels_text = tokenizer.apply_chat_template(
        conversation=prompts,
        add_generation_prompt=False,
        padding = True,
        tokenize=False,
    )
    labels = tokenizer(
        labels_text,
        padding="max_length",
        max_length = 256
    )['input_ids']
    for i in range(len(labels)):
        inputs.append(
            {
                'input_ids' : questions['input_ids'][i],
                'attention_mask' : questions['attention_mask'][i],
                'label' : labels[i]
            }
        )
    return inputs

# load the data
certain_file = f'{data_file}_certain.json'
uncertain_file = f'{data_file}_uncertain.json'
with open(certain_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
certain_data = preprocess_data(data)
with open(uncertain_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
uncertain_data = preprocess_data(data)
# print(f'the length of certain is {len(certain_data)}, the length of uncertain is {len(uncertain_data)}')
combine_data = certain_data + uncertain_data
tokenized_data = Dataset.from_list(tokenize_function(combine_data))


# fine tune
trainer = Trainer(
    model=model,  # model
    args=training_args,  # args
    train_dataset=tokenized_data,  # dataset
    # data_collator=data_collator, # data collator for padding
)

trainer.train()

# save the model
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model = PeftModel.from_pretrained(model, output_file)
# model.save_pretrained(output_file)

# test the data
output_file = 'dataset/test.json'
with open(output_file, 'w',encoding='utf-8') as f:
    json.dump(combine_data, f, indent=4, ensure_ascii=False)
print('finish fine tune!')