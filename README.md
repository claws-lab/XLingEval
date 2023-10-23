# XLingEval (Cross-Lingual Evaluation of LLMs)

# Paper Title: _Better to Ask in English: Cross-Lingual Evaluation of Large Language Models for Healthcare Queries_

<div align="center">
  <img src="static/img/Ask_Me_in_English_v2.png" width="400">
</div>

This is the project file from the paper **Better to Ask in English: Cross-Lingual Evaluation of Large Language 
Models for Healthcare Queries** ([arXiv Link](https://arxiv.org/abs/2310.13132))

XLingEval is an evaluation toolkit designed to assess the performance of large language models like GPT-3.5/4 and MedAlpaca in the context of medical queries across multiple languages. The toolkit focuses on three core metrics: correctness, consistency, and verifiability.

## Introduction

Large language models (LLMs) are transforming the ways the general public accesses and consumes information. Their influence is particularly pronounced in pivotal sectors like healthcare, where lay individuals are increasingly appropriating LLMs as conversational agents for everyday queries. While LLMs demonstrate impressive language understanding and generation proficiencies, concerns regarding their safety remain paramount in these high-stake domains. Moreover, the development of LLMs is disproportionately focused on English. It remains unclear how these LLMs perform in the context of non-English languages, a gap that is critical for ensuring equity in the real-world use of these systems.This paper provides a framework to investigate the effectiveness of LLMs as multi-lingual dialogue systems for healthcare queries. Our empirically derived framework XlingEval focuses on three fundamental criteria for evaluating LLM responses to naturalistic human-authored health-related questions: correctness, consistency, and verifiability. Through extensive experiments on four major global languages, including English, Spanish, Chinese, and Hindi, spanning three expert-annotated large health Q&A datasets, and through an amalgamation of algorithmic and human-evaluation strategies, we found a pronounced disparity in LLM responses across these languages, indicating a need for enhanced cross-lingual capabilities. We further propose XlingHealth, a cross-lingual benchmark for examining the multilingual capabilities of LLMs in the healthcare context. Our findings underscore the pressing need to bolster the cross-lingual capacities of these models, and to provide an equitable information ecosystem accessible to all.


## Installation

Create a new environment


```bash
conda create -n xlingeval python=3.9
conda activate xlingeval
```

Install all dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## XLingHeath Dataset

`XLingHealth_Dataset` folder inside the root repository contains the cross-lingual benchmarking versions for `HealthQA`, `LiveQA`, and `MedicationQA` datasets as Excel files under separate tabs for each of the four languages (English, Spanish, Chinese, and Hindi).


## Quick Start

### 1. Running Correctness Experiments

####  1.1 Evaluation using GPT-3.5

- To retrieve answers for questions using GPT-3.5 execute the following command in the root directory:

  ```bash
  python correctness/correctness_get_gpt_answer.py --dataset_path <path to the dataset> --model gpt-3.5-turbo
  ```
- To evaluate the quality between the ground-truth answer and the LLM answer, execute the following command in the root directory:

  ```bash
  python correctness/correctness_get_gpt_answer.py --dataset_path <path to the dataset> --model gpt-3.5-turbo
  ```
  

#### 1.2 Evaluation using MedAlpaca

- Retrieve answers from MedAlpaca using the following command in the root directory:
  ```bash
  python correctness/correctness/MedAlpaca/correctness_medalpaca_get_answers.py --dataset_path <path to the dataset> --model medalpaca-30b --batch_size 5
  ```
- Evaluate the quality between ground-truth answer and MedAlpaca LLM answer using GPT-3.5 execute the following command in the root directory:

  ```bash
  python correctness/correctness_get_gpt_answer.py --dataset_path <path to the dataset with MedAlpaca llm answers> --model gpt-3.5-turbo
  ```
    
### 2. Running Consistency Experiments

Run the following commands from the repository root directory `XLingEval/`. 

* Generate answers using GPT-3.5

  ```bash
  python consistency/consistency_get_gpt_answer.py --dataset <DATASET> --model <MODEL> --num_answers <NUM_ANSWERS>
  ```
  - `dataset`: select from `healthqa`, `liveqa`, `medicationqa`;
  - `model`: select from `gpt35`, `gpt4`, `medalpaca-7b`, `medalpaca-13b`, `medalpaca-30b`.  
  - `num_answers`: number of answers to generate for each question.

  For example

  ```bash
  python consistency/consistency_get_gpt_answer.py --dataset liveqa --model gpt35 --num_answers 10
  ```
  
* Translate the answers into English
  ```bash
  python consistency/translate.py --dataset <DATASET> --model <MODEL> --num_answers <NUM_ANSWERS>
  ```
  
* Evaluate the consistency metrics
* ```bash
  python consistency/consistency_answer_evaluation.py --dataset <DATASET> --model <MODEL> --num_answers <NUM_ANSWERS>
  ```
  
  For example
  
  ```bash
  python consistency/consistency_answer_evaluation.py --dataset liveqa --model gpt35
  ```
  
  The results will be saved in `outputs/consistency/`.



### 3. Running Verifiability Experiments

Run the following command from the repository root directory `XLingEval/`. Both GPT-3.5/4 and MedAlpaca models share the same code.

* Prompt the LLMs to generate answers

  ```bash
  python verifiability/verifiability_get_answer.py --dataset <DATASET> --model <MODEL>
  ```

  - `dataset`: select from `healthqa`, `liveqa`, `medicationqa`;
  - `model`: select from `gpt35`, `gpt4`, `medalpaca-7b`, `medalpaca-13b`, `medalpaca-30b`.

  For example, if you run experiments on LiveQA using the GPT-3.5 model:

  ```bash
  python verifiability/verifiability_get_answer.py --dataset liveqa --model gpt35
  ```

  By default, we run the experiments on all languages, including `English`, `Spanish`, `Chinese`, and `Hindi`. 

* Summarize the verifiability metrics

  ```bash
  python verifiability/verifiability_answer_evaluation.py --dataset <DATASET> --model <MODEL>
  ```

  For example, if you run experiments on LiveQA using the GPT-3.5 model:

  ```bash
  python verifiability/verifiability_answer_evaluation.py --dataset liveqa --model gpt35
  ```

  The results will be saved in `outputs/verifiability/`.

<div align="center">
  <img src="static/img/alpaca_doctor2.png" width="400">
</div>


## Repository Structure

### Correctness
- `const.py`: Constants used in the experiments.
- `correctness_get_gpt_answer.py`: Script to retrieve GPT-3.5-based answers for evaluation.
- `correctness_answer_evaluation.py`: Script to evaluate the correctness of llm-generated answers with the ground-truth using GPT-3.5.
- `setup.py`: Installation script.
- `utils_chatgpt.py`: Utilities for working with GPT-3.5 turbo using OpenAI API.
- `MedAlpaca/correctness_medalpaca_get_answers`: Script to retrieve answers from MedAlpaca model.

### Consistency
- `consistency_gpt.py` & `consistency_medalpaca.py`: Evaluate the consistency of answers from GPT and MedAlpaca models.
- `data_consistency.py`: Data handling for consistency evaluation.
- `prompts.py`: Pre-defined prompts for the experiments.
- `statistical_test.py`: Perform statistical tests on consistency metrics.
- `eval_consistency.py` & `eval_language_consistency.py`: Summarization scripts for consistency metrics.
- `language_consistency.py`: Evaluate language-specific consistency.

### Verifiability
- `prompts.py`: Pre-defined prompts for verifiability tests.
- `verifiability.py`: Main script for verifiability evaluation.
- `summarize_verifiability.py`: Summarize verifiability metrics.

### Data
- Houses datasets like `HealthQA`, `LiveQA`, and `MedicationQA` in Excel format.

### DataLoader
- `load_data.py`: Load and preprocess datasets for evaluation.

### Translate
- `translate_chatgpt.py`: Script to translate content using ChatGPT.

### Utils
- `metrics.py`: Evaluation metrics and utility functions.
- `utils_data.py`: Data handling utilities.
- `utils_misc.py`: Miscellaneous utility functions.

### Visual
- Scripts for visualizing data including line plots, heatmaps, and boxplots.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

