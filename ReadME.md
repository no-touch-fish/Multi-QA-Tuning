# **Pararel Dataset Processing Pipeline**
You need to first set up the environment.
```bash
pip install -r requirements.txt
```
This document outlines the step-by-step process for handling the **Pararel** dataset in our pipeline.

## **1. Preprocessing the Dataset**
Before training, the dataset needs to be preprocessed. The following command processes both the test and training sets:

```bash
# Process the Pararel test dataset
python template.py --data_path dataset/blank/pararel_test.json --save_path dataset/blank/processed_pararel_test.json --case blank --question_number 3

# Process the Pararel training dataset
python template.py --data_path dataset/blank/pararel_train.json --save_path dataset/blank/processed_pararel_train.json --case blank --question_number 3
```

---

## **2. Generating Model Output**
Once the dataset is processed, we generate model predictions using the following commands:

```bash
# Generate predictions for the Pararel test dataset
python generate_output.py --data_path dataset/blank/processed_pararel_test.json --save_path result/blank/pararel.json --case blank --generate_vllm --question_number 3 --gpu 0

# Generate predictions for the Pararel training dataset
python generate_output.py --data_path dataset/blank/processed_pararel_train.json --save_path result/blank/pararel.json --case blank --generate_vllm --question_number 3 --gpu 0
```

---

## **3. Comparing Model Output with Ground Truth**
To evaluate the model’s performance, compare its predictions against the ground truth labels:

```bash
python compare.py --data_path result/blank/pararel.json --case blank --question_number 3
```

---

## **4. Splitting the Dataset into Certain and Uncertain Cases**
To improve model robustness, we categorize the dataset into **certain** and **uncertain** instances:

```bash
python divide_dataset.py --data_path dataset/blank/processed_pararel_train.json --result result/blank/pararel.json --save_path dataset/blank/pararel_split/pararel --case blank
```

---

## **5. Fine-Tuning the Model**
To enhance model performance, fine-tune it using the **Pararel** dataset:

```bash
# Fine-tune using LLaMA3
python fine_tune.py --data_path dataset/blank/pararel_split/pararel --save_path models/blank/llama3_pararel --case blank --question_number 3 --gpu 0

# Fine-tune using Qwen
python fine_tune_Qwen.py --data_path dataset/blank/pararel_split/pararel --save_path models/blank/llama3_pararel --case blank --question_number 3 --gpu 0
```

---

## **6. Generating Output After Fine-Tuning**
After fine-tuning, we generate new predictions using the updated model:

```bash
python generate_output.py --data_path dataset/blank/processed_pararel_test.json --save_path fine_tune_result/blank/pararel.json --lora_model --lora_path models/blank/llama3_pararel --case blank --question_number 3 --gpu 0
```

---

## **7. Comparing Fine-Tuned Model Output with Ground Truth**
To assess the improvement, compare the fine-tuned model’s output:

```bash
python compare.py --data_path fine_tune_result/blank/pararel.json --case blank --question_number 3
```

---

## **8. Calculating AP Score**
To quantify the model’s reliability, compute the **AP (Average Precision) score**:

```bash
# AP Score for fine-tuned model
python calculate_ap.py --data_path fine_tune_result/blank/pararel.json --lora_model --lora_path models/blank/llama3_pararel --case blank --gpu 0
```

---
This pipeline ensures a systematic approach to processing, fine-tuning, and evaluating the **Pararel** dataset. 🚀
