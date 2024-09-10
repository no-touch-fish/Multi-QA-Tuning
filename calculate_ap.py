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
    default = 4,
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
df = pd.DataFrame(data)
questions = df['question'].tolist()
outputs = df['output'].tolist()
labels = df['label'].tolist()

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        logprobs=5,
    )
model = LLM(
        model=model_name, 
        enable_lora=True,
        max_num_seqs = 16,
    )

# function to divide the answer from the output
def divide_content(content):
    output_list = []
    index_1 = content.find('1:')
    index_2 = content.find('2:')
    index_3 = content.find('3:')
    index_end = content[index_3:].find('\n')
    if index_1 == -1 or index_2 == -1 or index_3 == -1:
        return None
    output_list.append(content[index_1+3:index_2-1])
    output_list.append(content[index_2+3:index_3-1])
    output_list.append(content[index_3+3:index_3 + index_end-1])
    return output_list

# function to get the prob of first index
def get_prob(logprobs):
    prob_dic = {' sure': 0, ' unsure': 0}
    for index,logprob in enumerate(logprobs):
        if index > 1:
            break
        for key in logprob:
            if logprob[key].decoded_token == ' sure':
                prob = math.exp(logprob[key].logprob)
                prob_dic[' sure'] = prob
            elif logprob[key].decoded_token == ' unsure':
                prob = math.exp(logprob[key].logprob)
                prob_dic[' unsure'] = prob
    return prob_dic

# function to get AP score
def calculate_scores(label,prob):
        p1, r1, t1 = precision_recall_curve(label, prob)
        ap_score = average_precision_score(label, prob)
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

        return ap_score

# get the prompt for Q-A pair and label for AP score
prompts = []
input_label = []
original_questions = []
additional_part = 'Are you sure you accurately answered the question based on your internal knowledge? I am'
for question, output,label in zip(questions, outputs,labels):
    question_list = divide_content(question)
    output_list = divide_content(output)
    if output_list == None:
        print('wrong answer format, not count!')
        continue
    for index in range(3):
        prompts.append(f'Question:{question_list[index]},Answer:{output_list[index]}.{additional_part}')
        original_questions.append(question_list[index])
        input_label.append(label[index])

results = model.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("lora_adapter", 1, lora_path)
        )
confidences = []
probs = []
for result in results:
        confidences.append(result.outputs[0].text)
        probs.append(get_prob(result.outputs[0].logprobs))
    
output_data = []
for index, (prompt,confidence,prob,label) in enumerate(zip(prompts,confidences,probs,input_label)):
    output_data.append({
        'question':question,
        'label': label,
        'confidence':confidence,
        'prob':prob
    })

# save the file to check the prompt
output_file = 'fine_tune_result/test.json'
with open(output_file, 'w') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"save the result to {output_file}")


# calculate the AP score
input_prob = []
for prob,label in zip(probs,input_label):
    input_prob.append(prob[' sure'])

    # input_prob.append(0.5*(prob[' sure'] + prob[' unsure']))

    # if label == 1:
    #     input_prob.append(prob[' sure'] - prob[' unsure'])
    # else:
    #     input_prob.append(prob[' unsure'] - prob[' sure'])

score = calculate_scores(input_label,input_prob)
print(f"Average Precision Score: {score}")











    



