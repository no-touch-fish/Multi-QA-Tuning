from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch
import json
import argparse
import os

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
    per_device_train_batch_size= 1, # batch size
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

def preprocess_data(data):
# apply templates to every three lines of original dataset
    combined_data = []
    for i in range(0, len(data), 3):
        if i+2 >= len(data):
            break
        combined_question = f'{template} 1: {data[i]["question"]} \n 2: {data[i+1]["question"]} \n 3:{data[i+2]["question"]}\n'
        combined_answer = f'{data[i]["answer"]} \n {data[i+1]["answer"]} \n {data[i+2]["answer"]}'
        combined_data.append({
            "question": combined_question, 
            "answer": combined_answer
        })
    return combined_data

def tokenize_function(example):
    MAX_LENGTH = 256    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['answer']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

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
data = Dataset.from_list(combine_data)
tokenized_data = data.map(tokenize_function)
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
# save_data = []
# for i, example in enumerate(tokenized_data):
#         # 解码 input_ids 和 labels
#         input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
#         label_text = tokenizer.decode(example['labels'], skip_special_tokens=True)
#         save_data.append({
#             'question': input_text,
#             'answer' : label_text
#         })

# output_file = 'dataset/test.json'
# with open(output_file, 'w',encoding='utf-8') as f:
#     json.dump(save_data, f, indent=4, ensure_ascii=False)
print('finish fine tune!')