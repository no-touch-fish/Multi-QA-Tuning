import pandas as pd
import torch
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
import os
import numpy as np
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="""parameter"""
)
parser.add_argument(
    "--data_path",
    type = str,
    default = None,
    help = "Path to the original dataset used.",
)
parser.add_argument(
    "--gpu",
    type = str,
    help = "which gpu to use",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 16,
    help = "the batch size for the input",
)
parser.add_argument(
    "--case",
    type = str,
    default = 'choice',
    help = "choice or blank",
)
parser.add_argument(
    '--lora_path',
    type = str,
    default = None,
    help = "the path to the lora model",
)
parser.add_argument(
    '--lora_model', 
    action='store_true', 
    help="Generate using lora fine-tuned model",
)
parser.add_argument(
    '--MTI', 
    action='store_true', 
    help="new dataset",
)
parser.add_argument(
    '--SINGLE', 
    action='store_true', 
    help="question number = 1 or not",
)

args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
case = args.case
batch_size = args.batch_size
lora_path = args.lora_path

with open(data_file, 'r',encoding='utf-8') as file:
    data = json.load(file)
# df = pd.DataFrame(data).head(100)
df = pd.DataFrame(data)
questions = df['question'].tolist()
outputs = df['output'].tolist()
labels = df['label'].tolist()

# model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'Qwen/Qwen2-7B-Instruct'
model_name = 'meta-llama/Llama-3.2-3B-Instruct'

sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=5,
    )
model = LLM(
        model=model_name, 
        enable_lora=True,
        max_num_seqs = 16,
        gpu_memory_utilization=0.9,
        max_model_len = 2048,
    )

def divide_MTI_question(content):
    output_list = []
    # deal with context
    if content.find('Only response to the content below') != -1:
        content = content[content.find('Only response to the content below')+8:]
    index_1 = content.find('1:')
    index_2 = content.find('2:')
    index_3 = content.find('3:')
    index_end = content[index_3:].find('\n')
    index_context = content.find('###Context')
    if index_1 == -1 or index_2 == -1 or index_3 == -1:
        return None
    output_list.append(content[index_1+3:index_2-1]+content[index_context:])
    output_list.append(content[index_2+3:index_3-1]+content[index_context:])
    if index_end == -1:
        output_list.append(content[index_3+3:]+content[index_context:])
    else:
        output_list.append(content[index_3+3:index_3 + index_end-1]+content[index_context:])
    return output_list

# function to divide MTI dataset output
def divide_MTI(content):
    output_list = []
    # deal with context
    if content.find('Only response to the content below') != -1:
        content = content[content.find('Only response to the content below')+8:]
    index_1 = content.find('<task1>')
    index_2 = content.find('<task2>')
    index_3 = content.find('<task3>')
    if index_1 == -1 or index_2 == -1 or index_3 == -1:
        return divide_content(content)
    index_1_end = content.find('<task1/>')
    if index_1_end == -1:
        index_1_end = content[index_1:].find('\n')
        output_list.append(content[index_1:index_1 + index_1_end])
    else:
        output_list.append(content[index_1:index_1_end+8])
    
    index_2_end = content.find('<task2/>')
    if index_2_end == -1:
        index_2_end = content[index_2:].find('\n')
        output_list.append(content[index_2:index_2 + index_2_end])
    else:
        output_list.append(content[index_2:index_2_end+8])
    
    index_3_end = content.find('<task3/>')
    if index_3_end == -1:
        index_3_end = content[index_3:].find('\n')
        if index_3_end == -1:
            output_list.append(content[index_3:])
        else:
            output_list.append(content[index_3:index_3 + index_3_end])
    else:
        output_list.append(content[index_3:index_3_end+8])
    return output_list

# function to divide the content
def divide_content(content):
    output_list = []
    # deal with context
    if content.find('Only response to the content below') != -1:
        content = content[content.find('Only response to the content below')+8:]

    index_1 = content.find('1:')
    index_2 = content.find('2:')
    index_3 = content.find('3:')
    index_end = content[index_3:].find('\n')
    if index_1 == -1 or index_2 == -1 or index_3 == -1:
        return None
    output_list.append(content[index_1+3:index_2-1])
    output_list.append(content[index_2+3:index_3-1])
    if index_end == -1:
        output_list.append(content[index_3+3:])
    else:
        output_list.append(content[index_3+3:index_3 + index_end-1])
    return output_list

def divide_content_2(content):
    output_list = content.split('\n')
    # print(output_list)
    return output_list

# function to get the prob of first index
def get_prob(logprobs):
    prob_dic = {' sure': 0, ' unsure': 0}
    for index,logprob in enumerate(logprobs):
        if index > 1:
            break
        for key in logprob:
            # print(logprob[key].decoded_token)
            if logprob[key].decoded_token == ' sure':
                prob = math.exp(logprob[key].logprob)
                # print(f'sure prob:{prob}')
                prob_dic[' sure'] += prob
            if logprob[key].decoded_token == ' confident':
                prob = math.exp(logprob[key].logprob)
                # print(f'confident prob:{prob}')
                # prob_dic[' sure'] = max(prob_dic[' sure'], prob)
                prob_dic[' sure'] += prob
            if logprob[key].decoded_token == ' unsure':
                prob = math.exp(logprob[key].logprob)
                # print(f'unsure prob:{prob}')
                prob_dic[' unsure'] += prob
            if logprob[key].decoded_token == ' not':
                prob = math.exp(logprob[key].logprob)
                # print(f'not prob:{prob}')
                # prob_dic[' unsure'] = max(prob_dic[' unsure'], prob)
                prob_dic[' unsure'] += prob
    if prob_dic[' sure'] == 0 and prob_dic[' unsure'] == 0:
        prob_dic[' sure'] = 0.5
        prob_dic[' unsure'] = 0.5
    else:
        sure_prob = prob_dic[' sure'] / (prob_dic[' sure'] + prob_dic[' unsure'])
        prob_dic[' sure'] = sure_prob
        prob_dic[' unsure'] = 1 - sure_prob
    return prob_dic

def precison_recall_curve(label,prob):
    print('start the curve making')
    p1, r1, t1 = precision_recall_curve(label, prob)
    
    # print the curve
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # figsize 参数控制图像的整体大小
    # first plot
    axs[0].plot(p1,r1)
    axs[0].set_title("precision_recall_curve")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    # second plot
    axs[1].plot(t1, r1[:-1])
    axs[1].set_title("recall_threhold_curve")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('Threhold')
    axs[1].set_ylabel('Recall')   
    # third plot
    axs[2].plot(t1, p1[:-1])
    axs[2].set_title("precision_threhold_curve")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel('Threhold')
    axs[2].set_ylabel('Precision')
    # set the name of the whole figure
    fig.suptitle(data_file[16:], fontsize=16)
    # avoid overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # save the figure
    plt.savefig("multiple_plots.png")
    return 1

# function to deal with CoQA
def CoQA(questions,outputs,labels,storys):
    prompts = []
    input_label = []
    original_questions = []
    additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? I am'
    for question, output, label, story in zip(questions, outputs, labels, storys):
        question_list = divide_content(question)
        output_list = divide_content(output)
        if output_list == None:
            print('wrong answer format, not count!')
            continue
        for index in range(3):
            prompts.append(f'{story}\nQuestion:{question_list[index]},Answer:{output_list[index]}.{additional_part}')
            original_questions.append(question_list[index])
            input_label.append(label[index])
    if args.lora_model:
        results = model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
                lora_request=LoRARequest("lora_adapter", 1, lora_path)
                )
    else:
        results = model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
                )
    confidences = []
    probs = []
    for result in results:
            confidences.append(result.outputs[0].text)
            probs.append(get_prob(result.outputs[0].logprobs))
        
    output_data = []
    for index, (prompt,confidence,prob,label) in enumerate(zip(prompts,confidences,probs,input_label)):
        output_data.append({
            'question': prompt,
            'label': label,
            'confidence':confidence,
            'prob':prob
        })

    # save the file to check the prompt
    output_file = 'fine_tune_result/test.json'
    with open(output_file, 'w') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"save the result to {output_file}")
    return probs, input_label

# get the prompt for Q-A pair and label for AP score
if 'story' in df.columns:
    storys = df['story'].tolist()
    probs,input_label = CoQA(questions,outputs,labels,storys)
else:
    prompts = []
    input_label = []
    original_questions = []
    additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? I am'
    for question, output,label in zip(questions, outputs,labels):
        if args.MTI:
            question_list = divide_MTI_question(question)
            if args.lora_model:
                output_list = divide_MTI(output)
            else:
                output_list = divide_MTI(output)
        else:
            question_list = divide_content(question)
            output_list = divide_content(output)
        if output_list == None:
            output_list = divide_content_2(output)
            if output_list == None or len(output_list) != 3:
                # print(f'wrong answer format ({output}), not count!')
                print(f'wrong answer format, not count!')
                continue
        for index in range(3):
            if args.MTI:
                prompt = ''
                for i in range(index):
                    if prompt == '':
                        prompt = 'Context:'
                    prompt += f'{question_list[i]}\nAnswer:{output_list[i]}\n'
                prompts.append(f'{prompt}Question:{question_list[index]}\nAnswer:{output_list[index]}.{additional_part}')
                original_questions.append(question_list[index])
                input_label.append(label[index])
            else:
                prompts.append(f'Question:{question_list[index]}\nAnswer:{output_list[index]}.{additional_part}')
                original_questions.append(question_list[index])
                input_label.append(label[index])

    if args.lora_model:
        results = model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
                lora_request=LoRARequest("lora_adapter", 1, lora_path)
                )
    else:
        results = model.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
                )
    confidences = []
    probs = []
    logprobs = []
    for result in results:
            confidences.append(result.outputs[0].text)
            probs.append(get_prob(result.outputs[0].logprobs))
            tmp = result.outputs[0].logprobs[0]
            logprob = []
            for key in tmp:
                logprob.append({tmp[key].decoded_token : math.exp(tmp[key].logprob)})
            logprobs.append(logprob)
        
    output_data = []
    for index, (prompt,confidence,logprob,prob,label) in enumerate(zip(prompts,confidences,logprobs,probs,input_label)):
        # print(f'logprob:{logprob}')
        output_data.append({
            'question': prompt,
            'label': label,
            'logprob': logprob,
            'confidence':confidence,
            'prob':prob
        })

    # save the file to check the prompt
    output_file = 'fine_tune_result/test.json'
    with open(output_file, 'w') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"save the result to {output_file}")

# calculate the accuracy
input_prob = []
for prob in probs:
    input_prob.append(prob[' sure'])

total = 0
correct = 0
for label, prob in zip(input_label,input_prob):
    if prob > 0.5:
        total += 1
        if label == 1:
            correct += 1
if total != 0:
    print(f"accuracy: {correct/total}")

# calculate the AP score
a = precision_recall_curve(input_label,input_prob)

sure_score = average_precision_score(input_label,input_prob)
print(f"Average Precision Score for Sure: {sure_score}")

# calculate the Calibration Error
M = 10  # num of interval
bins = np.linspace(0, 1, M + 1) 
bucket_samples = [[] for _ in range(M)] 
ECE = 0
MCE = 0
n = len(input_label)
# put to bucket
for prob, label in zip(input_prob, input_label):
    bucket_index = min(np.digitize(prob, bins) - 1, M - 1)
    bucket_samples[bucket_index].append((prob, label))
for m, samples in enumerate(bucket_samples):
    if len(samples) == 0: 
        continue
    probs = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    conf_m = np.mean(probs)  # average confidence
    acc_m = np.mean(labels)  # real accuracy
    error_m = abs(conf_m - acc_m)  # error of the bucket
    ECE += len(samples) / n * error_m
    MCE = max(MCE, error_m)
print(f'ECE is: {ECE}, MCE is: {MCE}')

# calculate the F1 score
threshold = 0.5
prob_f1 = []
for p in input_prob:
    if p > threshold:
        prob_f1.append(1)
    else:
        prob_f1.append(0)

precision = precision_score(input_label, prob_f1)
recall = recall_score(input_label, prob_f1)
f1 = f1_score(input_label, prob_f1)

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")













    



