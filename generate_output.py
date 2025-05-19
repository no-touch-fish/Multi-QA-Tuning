import pandas as pd
from transformers import AutoTokenizer
import torch
import json
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
import argparse
import os
import math

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
    help = "Path to the save dir",
)
parser.add_argument(
    "--case",
    type = str,
    default = 'choice',
    help = "choice or blank",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 16,
    help = "the batch size for the input",
)
parser.add_argument(
    '--generate_vllm', 
    action='store_true', 
    help="Generate using vllm",
)
parser.add_argument(
    '--lora_model', 
    action='store_true', 
    help="Generate using lora fine-tuned model",
)
parser.add_argument(
    '--lora_path',
    type = str,
    default = None,
    help = "the path to the lora model",
)
parser.add_argument(
    "--question_number",
    type = int,
    default = 3,
    help = "the number for how many questions combine together",
)

# vllm setting
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]= "spawn"

args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = args.save_path
case = args.case
batch_size = args.batch_size
lora_path = args.lora_path
question_number = args.question_number

# load dataset
with open(data_file, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
questions = df['question'].tolist()
answers = df['answer'].tolist()

# load model
# model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'Qwen/Qwen2-7B-Instruct'
model_name = 'Qwen/Qwen2.5-14B-Instruct'
# model_name = "microsoft/Phi-3.5-mini-instruct"
# model_name = 'meta-llama/Llama-3.2-3B-Instruct'

# put the input into format
def get_generate_input(questions,model_name):
    inputs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
            [
                {
                    "role": "user",
                    "content": question,
                }
            ] for question in questions
        ]

    inputs = tokenizer.apply_chat_template(
        conversation=prompts,
        add_generation_prompt=True,
    )
    return inputs

# generate using vllm framework
def generate_vllm(inputs,model_name,batch_size):
    llm = LLM(
        model= model_name,
        max_num_seqs = batch_size,
        gpu_memory_utilization=0.9,
        max_model_len = 4096,
        )
    sampling_params = SamplingParams(
        temperature = 0.0, 
        top_p = 1,
        max_tokens=1024,
        )
    results = llm.generate(
        prompt_token_ids = inputs,
        sampling_params = sampling_params,
        use_tqdm = True,
        )
    generation = []
    for result in results:
        generation.append(result.outputs[0].text)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        logprobs=5,
        )
    confidence = []
    probs = []
    if question_number == 1:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure.'
    elif question_number == 2:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure.'
    elif question_number == 3:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure \n3: I am sure/unsure.'
    elif question_number == 4:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure \n3: I am sure/unsure \n4: I am sure/unsure.'
    elif question_number == 5:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure \n3: I am sure/unsure \n4: I am sure/unsure \n5: I am sure/unsure.'
    prompts = []
    for question, output in zip(questions,generation):
        prompts.append(f'Question:{question}\nAnswer:{output}.{additional_part}')
    inputs = get_generate_input(prompts,model_name)
    results = llm.generate(
        prompt_token_ids=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
        )
    for result in results:
        confidence.append(result.outputs[0].text)
        probs.append(get_prob(result.outputs[0].logprobs))
    return probs,confidence, generation

# generate with Lora fine-tuning, using vllm framework
def generate_lora(questions,model_name,batch_size):
    lora_file = lora_path
    inputs = get_generate_input(questions,model_name)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        )
    model = LLM(
        model=model_name, 
        enable_lora=True,
        max_num_seqs = batch_size,
        gpu_memory_utilization=0.9,
        max_model_len = 4096,
        )
    results = model.generate(
        prompt_token_ids=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("lora_adapter", 1, lora_file)
        )
    generation = []
    for result in results:
        generation.append(result.outputs[0].text)
    # get the confidence and logprob
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        logprobs=5,
        )
    confidence = []
    probs = []
    if question_number == 1:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure.'
    elif question_number == 2:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure.'
    elif question_number == 3:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure \n3: I am sure/unsure.'
    elif question_number == 4:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure \n3: I am sure/unsure \n4: I am sure/unsure.'
    elif question_number == 5:
        additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? Answer in following format: 1: I am sure/unsure \n2: I am sure/unsure \n3: I am sure/unsure \n4: I am sure/unsure \n5: I am sure/unsure.'
    prompts = []
    for question, output in zip(questions,generation):
        prompts.append(f'Question:{question}\nAnswer:{output}.{additional_part}')
    inputs = get_generate_input(prompts,model_name)
    results = model.generate(
        prompt_token_ids=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("lora_adapter", 1, lora_file)
        )
    for result in results:
        confidence.append(result.outputs[0].text)
        probs.append(get_prob(result.outputs[0].logprobs))
    return probs,confidence, generation

def get_prob(logprobs):
    prob_list = []
    for index,logprob in enumerate(logprobs):
        if index > 20:
            break
        for key in logprob:
            if logprob[key].decoded_token == ' sure':
                prob = math.exp(logprob[key].logprob)
                prob_list.append({' sure':prob})
                break
            elif logprob[key].decoded_token == ' unsure':
                prob = math.exp(logprob[key].logprob)
                prob_list.append({' unsure':prob})
    return prob_list

def main():

    # generate the output
    if args.generate_vllm:
        inputs = get_generate_input(questions,model_name)
        probs,confidence, generations = generate_vllm(inputs,model_name,batch_size)
    elif args.lora_model:
        probs,confidence, generations = generate_lora(questions,model_name,batch_size)

    # get the output
    output_data = []
    if case == 'choice':
        original_options = df['original_options'].tolist()
        for question, generation, answer,original_option in zip(questions, generations, answers, original_options):
            output_data.append({
                'question': question, 
                'original_options':original_option,
                'answer': answer,
                'output': generation
            })
    elif case == 'blank':
        for question, generation, answer in zip(questions, generations, answers):
            output_data.append({
                'question': question, 
                'answer': answer,
                'output': generation
                })

    # generate the confidence
    for entry,item,prob in zip(output_data,confidence,probs):
        entry['confidence'] = item
        # entry['prob'] = prob
    if 'story' in df.keys():
        storys = df['story'].tolist()
        for entry,story in zip(output_data,storys):
            entry['story'] = story

    # save the output
    with open(output_file, 'w') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"save the result to {output_file}")

if __name__ == '__main__':
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    main()

