# XLingEval


<div align="center">
  <img src="static/img/alpaca_doctor1.png" width="400">
</div>

This is the project file from the paper **Ask Me in English Instead**: Cross-Lingual Evaluation of Large Language Models for Healthcare Queries ([arXiv]())

XLingEval is an evaluation toolkit designed to assess the performance of large language models like GPT-3.5/4 and MedAlpaca in the context of medical queries across multiple languages. The toolkit focuses on three core metrics: correctness, consistency, and verifiability.

## Introduction

Large language models (LLMs) are transforming the ways the general public accesses and consumes information. Their influence is particularly pronounced in pivotal sectors like healthcare, where lay individuals are increasingly appropriating LLMs as conversational agents for everyday queries. While LLMs demonstrate impressive language understanding and generation proficiencies, concerns regarding their safety remain paramount in these high-stake domains. Moreover, the development of LLMs is disproportionately focused on English. It remains unclear how these LLMs perform in the context of non-English languages, a gap that is critical for ensuring equity in the real-world use of these systems.This paper provides a framework to investigate the effectiveness of LLMs as multi-lingual dialogue systems for healthcare queries. Our empirically-derived framework XlingEval focuses on three fundamental criteria for evaluating LLM responses to naturalistic human-authored health-related questions: correctness, consistency, and verifiability. Through extensive experiments on four major global languages, including English, Spanish, Chinese, and Hindi, spanning three expert-annotated large health Q&A datasets, and through an amalgamation of algorithmic and human-evaluation strategies, we found a pronounced disparity in LLM responses across these languages, indicating a need for enhanced cross-lingual capabilities. We further propose XlingHealth, a cross-lingual benchmark for examining the multilingual capabilities of LLMs in the healthcare context. Our findings underscore the pressing need to bolster the cross-lingual capacities of these models, and to provide an equitable information ecosystem accessible to all.


## Installation

Install all dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## XLingHeath Benchmark

`XLingHealth` folder inside the root repository contains the cross-lingual benchmarking versions for `HealthQA`, `LiveQA`, and `MedicationQA` datasets as tsv files. Each dataset contains the following columns:

\[question,answer,translated_question_Hindi,translated_answer_Hindi,translated_question_Chinese,translated_answer_Chinese,translated_question_Spanish,translated_answer_Spanish\]

Where `question` and `answer` columns are obtained from the original referred datasets.

## Original Datasets

The original datasets used for constructing the **XLingHealth** benchmark can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1iZOhWXVNHGQXqPnGTQJMlaDQVznIRHci?usp=share_link). This include `HealthQA.xlsx`, `LiveQA.xlsx`, and `MedicationQA.xlsx`.

## Quick Start

**To evaluate correctness using XLingEval:**

* To get answers for questions using GPT-3.5 execute:

```bash
python correctness/correctness_get_gpt_answer.py --dataset_path <path to the dataset> --username <enter your username> --model gpt-35-turbo
```
* To evaluate the ground-truth answer and LLM answer, execute:

```bash
python correctness/correctness_get_gpt_answer.py --dataset_path <path to the dataset> --username <enter your username> --model gpt-35-turbo
```

**To evaluate verifiability using XLingEval, execute the following command in the root directory. Take :**

```bash
python verifiability/verifiability.py --dataset liveqa --model gpt35
```

- `dataset`: select from `healthqa`, `liveqa`, `medicationqa`;
- `model`: select from `gpt35`, `gpt4`, `medalpaca`.

By default, we run the experiments on all languages, including `English`, `Spanish`, `Chinese`, and `Hindi`. 

<div align="center">
  <img src="static/img/alpaca_doctor2.png" width="400">
</div>


## Repository Structure

### Correctness_Experiment_Files
- `const.py`: Constants used in the experiments.
- `correctness_get_gpt_answer.py`: Retrieve GPT-3.5-based answers for evaluation.
- `correctness_answer_evaluation.py`: Script to evaluate the correctness of llm-generated answers with the ground-truth using GPT-3.5.
- `setup.py`: Installation script.
- `utils_chatgpt.py`: Utilities for working with GPT-3.5 turbo using OpenAI api.

### Consistency
- `consistency_gpt.py` & `consistency_medalpaca.py`: Evaluate the consistency of answers from GPT and MedAlpaca models.
- `data_consistency.py`: Data handling for consistency evaluation.
- `prompts.py`: Pre-defined prompts for the experiments.
- `statistical_test.py`: Perform statistical tests on consistency metrics.

### Data
- Houses datasets like `HealthQA`, `LiveQA`, and `MedicationQA` in Excel format.

### DataLoader
- `load_data.py`: Load and preprocess datasets for evaluation.

### Eval
- `eval_consistency.py` & `eval_language_consistency.py`: Summarization scripts for consistency metrics.
- `language_consistency.py`: Evaluate language-specific consistency.
- `summarize_verifiability.py`: Summarize verifiability metrics.

### Translate
- `translate_chatgpt.py`: Script to translate content using ChatGPT.

### Utils
- `metrics.py`: Evaluation metrics and utility functions.
- `utils_data.py`: Data handling utilities.
- `utils_misc.py`: Miscellaneous utility functions.

### Verifiability
- `prompts.py`: Pre-defined prompts for verifiability tests.
- `verifiability.py`: Main script for verifiability evaluation.

### Visual
- Scripts for visualizing data including line plots, heatmaps, and boxplots.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

