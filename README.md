<div align="center">

# 🌐 XLingEval

### *Better to Ask in English: Cross-Lingual Evaluation of Large Language Models for Healthcare Queries*

[![Web Conference 2024](https://img.shields.io/badge/Web_Conference_2024-Oral-7B3FE4.svg)](https://doi.org/10.1145/3589334.3645643)
[![arXiv](https://img.shields.io/badge/arXiv-2310.13132-b31b1b.svg)](https://arxiv.org/abs/2310.13132)
[![Website](https://img.shields.io/badge/🌐_Project-Website-2EA44F.svg)](https://claws-lab.github.io/XLingEval/)
[![HF Dataset](https://img.shields.io/badge/🤗_Hugging_Face-XLingHealth-FFD21E.svg)](https://huggingface.co/datasets/claws-lab/XLingHealth)
[![Video](https://img.shields.io/badge/▶_YouTube-Talk-FF0000.svg)](https://www.youtube.com/watch?v=pmEafw5ZOPg)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3589334.3645643-1f6feb.svg)](https://doi.org/10.1145/3589334.3645643)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/claws-lab/XLingEval?style=social)](https://github.com/claws-lab/XLingEval/stargazers)

[Yiqiao Jin](https://ahren09.github.io/)¹\*, [Mohit Chandra](https://mohit3011.github.io/)¹\*, [Gaurav Verma](https://gaurav22verma.github.io/)¹, [Yibo Hu](https://itm.iit.edu/hu/)¹, [Munmun De Choudhury](http://www.munmund.net/)¹, [Srijan Kumar](https://faculty.cc.gatech.edu/~srijan/)¹

¹ Georgia Institute of Technology

<sub>\* Equal contribution</sub>

<img src="static/img/Ask_Me_in_English_v2.png" width="400">

</div>

---

## 📖 Abstract

Large language models (LLMs) are transforming the ways the general public accesses and consumes information. Their influence is particularly pronounced in pivotal sectors like healthcare, where lay individuals are increasingly appropriating LLMs as conversational agents for everyday queries. While LLMs demonstrate impressive language understanding and generation proficiencies, concerns regarding their safety remain paramount in these high-stake domains. Moreover, the development of LLMs is disproportionately focused on English. It remains unclear how these LLMs perform in the context of non-English languages, a gap that is critical for ensuring equity in the real-world use of these systems. This paper provides a framework to investigate the effectiveness of LLMs as multi-lingual dialogue systems for healthcare queries. Our empirically-derived framework **XLingEval** focuses on three fundamental criteria for evaluating LLM responses to naturalistic human-authored health-related questions: **correctness, consistency, and verifiability**. Through extensive experiments on four major global languages — English, Spanish, Chinese, and Hindi — spanning three expert-annotated large health Q&A datasets, and through an amalgamation of algorithmic and human-evaluation strategies, we found a pronounced disparity in LLM responses across these languages, indicating a need for enhanced cross-lingual capabilities. We further propose **XLingHealth**, a cross-lingual benchmark for examining the multilingual capabilities of LLMs in the healthcare context.

## ✨ Highlights

- 🌐 **First cross-lingual healthcare benchmark** — XLingHealth covers **4 major world languages × 3 expert-annotated Q&A datasets** (HealthQA, LiveQA, MedicationQA).
- 📉 **Pronounced English bias quantified** — GPT-3.5 produces **18.12% fewer** comprehensive answers and is **5.82× more likely** to give an incorrect response in non-English languages.
- ⚖️ **Three-axis evaluation framework** — XLingEval unifies **correctness**, **consistency**, and **verifiability**, combining algorithmic metrics with expert human evaluation.
- 🔬 **Multi-model coverage** — Validated across **GPT-3.5**, **GPT-4**, and the open-source **MedAlpaca** family (7B / 13B / 30B).
- 🌍 **Steep degradation in under-represented languages** — Semantic consistency drops **9.1% (es) / 28.3% (zh) / 50.5% (hi)** vs English; verifiability Macro-F1 drops up to **23.4% (hi)**.
- 🧰 **Generalizable beyond healthcare** — The same correctness / consistency / verifiability lens applies to legal, financial, and educational dialogue.

## 🧭 The XLingEval Framework

XLingEval evaluates LLM responses along three healthcare-critical axes. Each axis combines automated metrics with human evaluation by medical annotators across all four languages.

| Axis | What it measures | Key metrics |
| :--- | :--- | :--- |
| ✅ **Correctness** | Whether the LLM's answer matches expert ground-truth | LLM-judge comparative analysis (CoT prompting), human evaluation |
| 🔁 **Consistency** | Whether the LLM gives stable answers under sampling | n-gram & length (surface), BERTScore & SBERT (semantic), LDA / HDP (topic) |
| 🔎 **Verifiability** | Whether the LLM can authenticate medical claims | Macro-Precision, Macro-Recall, Macro-F1, Accuracy, AUC |

## 🌍 Supported Languages

| Language | Code | Role | Translation source |
| :--- | :---: | :--- | :--- |
| 🇬🇧 English | `en` | Baseline | Native |
| 🇪🇸 Spanish | `es` | Cross-lingual eval | MT + human verification |
| 🇨🇳 Simplified Chinese | `zh` | Cross-lingual eval | MT + human verification |
| 🇮🇳 Hindi | `hi` | Cross-lingual eval | MT + human verification |

## 🤖 Supported Models

| Family | Variants | Access |
| :--- | :--- | :--- |
| **GPT** | `gpt-3.5-turbo`, `gpt-4` | OpenAI API |
| **MedAlpaca** | `medalpaca-7b`, `medalpaca-13b`, `medalpaca-30b` | Open source (HF) |

## 📊 XLingHealth Dataset

The `XLingHealth_Dataset/` folder in the repository root contains the cross-lingual benchmark versions of `HealthQA`, `LiveQA`, and `MedicationQA` as Excel files, with separate tabs for each of the four languages (English, Spanish, Chinese, Hindi).

> 🤗 The dataset is also published on Hugging Face: **[claws-lab/XLingHealth](https://huggingface.co/datasets/claws-lab/XLingHealth)**.

| Dataset | #Examples | #Words (Q) | #Words (A) |
| :--- | :---: | :---: | :---: |
| **HealthQA** | 1,134 | 7.72 ± 2.41 | 242.85 ± 221.88 |
| **LiveQA** | 246 | 41.76 ± 37.38 | 115.25 ± 112.75 |
| **MedicationQA** | 690 | 6.86 ± 2.83 | 61.50 ± 69.44 |

- `#Words (Q)` and `#Words (A)` are the average word counts in the questions and ground-truth answers respectively.
- In **HealthQA**, each question is paired with **1 positive** and **9 negative** answers — total **11,340** examples.
- **LiveQA** and **MedicationQA** do not provide negatives; we sample **4 negatives** per question, yielding totals of **1,230** and **3,450** examples respectively.

## 🚀 Installation

Create a new conda environment:

```bash
conda create -n xlingeval python=3.9
conda activate xlingeval
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Correctness Experiments

#### 1.1 Evaluation using GPT-3.5

Retrieve answers from GPT-3.5:

```bash
python correctness/correctness_get_gpt_answer.py \
    --dataset_path <path to the dataset> \
    --model gpt-3.5-turbo
```

Evaluate the quality between the ground-truth and the LLM answer:

```bash
python correctness/correctness_answer_evaluation.py \
    --dataset_path <path to the dataset> \
    --model gpt-3.5-turbo
```

#### 1.2 Evaluation using MedAlpaca

Retrieve answers from MedAlpaca:

```bash
python correctness/MedAlpaca/correctness_medalpaca_get_answers.py \
    --dataset_path <path to the dataset> \
    --model medalpaca-30b \
    --batch_size 5
```

Evaluate the MedAlpaca answers using GPT-3.5 as a judge:

```bash
python correctness/correctness_answer_evaluation.py \
    --dataset_path <path to the dataset with MedAlpaca llm answers> \
    --model gpt-3.5-turbo
```

### 2. Consistency Experiments

Run all commands from the repository root `XLingEval/`.

* Generate answers with multiple samplings:

  ```bash
  python consistency/consistency_get_gpt_answer.py \
      --dataset <DATASET> --model <MODEL> --num_answers <NUM_ANSWERS>
  ```

  - `dataset`: `healthqa` · `liveqa` · `medicationqa`
  - `model`: `gpt35` · `gpt4` · `medalpaca-7b` · `medalpaca-13b` · `medalpaca-30b`
  - `num_answers`: number of samples per question

  Example:

  ```bash
  python consistency/consistency_get_gpt_answer.py \
      --dataset liveqa --model gpt35 --num_answers 10
  ```

* Translate the sampled answers back into English:

  ```bash
  python consistency/translate.py \
      --dataset <DATASET> --model <MODEL> --num_answers <NUM_ANSWERS>
  ```

* Evaluate consistency metrics:

  ```bash
  python consistency/consistency_answer_evaluation.py \
      --dataset <DATASET> --model <MODEL> --num_answers <NUM_ANSWERS>
  ```

  Results are written to `outputs/consistency/`.

### 3. Verifiability Experiments

Both GPT-3.5/4 and MedAlpaca share the same code path.

* Prompt the LLM to verify each (question, answer) pair:

  ```bash
  python verifiability/verifiability_get_answer.py \
      --dataset <DATASET> --model <MODEL>
  ```

  By default, all four languages (`en`, `es`, `zh`, `hi`) are evaluated.

* Summarize verifiability metrics:

  ```bash
  python verifiability/verifiability_answer_evaluation.py \
      --dataset <DATASET> --model <MODEL>
  ```

  Results are written to `outputs/verifiability/`.

<div align="center">
  <img src="static/img/alpaca_doctor2.png" width="400">
</div>

## 🗂️ Repository Structure

```
XLingEval/
├── correctness/        # Correctness pipeline (GPT-3.5/4 & MedAlpaca)
├── consistency/        # Consistency pipeline (sampling, translation, scoring)
├── verifiability/      # Verifiability pipeline (claim authentication)
├── translate/          # Translation utilities (ChatGPT-based)
├── dataloader/         # Dataset loading & preprocessing
├── utils/              # Metrics, data utilities, miscellaneous helpers
├── visual/             # Plots: line, heatmap, boxplot
├── XLingHealth_Dataset/  # Cross-lingual benchmark Excel files
├── outputs/            # Experiment outputs (correctness, consistency, verifiability)
├── static/, media/     # Project page assets
└── index.html          # Project website (open in browser)
```

### Module highlights

- **`correctness/`** — `correctness_get_gpt_answer.py`, `correctness_answer_evaluation.py`, `MedAlpaca/correctness_medalpaca_get_answers.py`
- **`consistency/`** — `consistency_get_gpt_answer.py`, `translate.py`, `consistency_answer_evaluation.py`, `statistical_test.py`
- **`verifiability/`** — `verifiability_get_answer.py`, `verifiability_answer_evaluation.py`, `prompts.py`
- **`utils/`** — `metrics.py`, `utils_data.py`, `utils_misc.py`
- **`visual/`** — line plots, heatmaps, and boxplots used in the paper

## 📝 Citation

If you find XLingEval or XLingHealth useful in your research, please cite:

```bibtex
@inproceedings{jin2024better,
  title={Better to ask in english: Cross-lingual evaluation of large language models for healthcare queries},
  author={Jin, Yiqiao and Chandra, Mohit and Verma, Gaurav and Hu, Yibo and De Choudhury, Munmun and Kumar, Srijan},
  booktitle={Proceedings of the ACM Web Conference 2024},
  pages={2627--2638},
  year={2024}
}
```

## 🙏 Acknowledgements

This work was supported in part by NSF (CNS-2154118, ITE-2137724, ITE-2230692, CNS-2239879), DARPA (HR00112290102, subcontract PO70745), CDC, and Microsoft. We thank our medical annotators for the cross-lingual evaluation effort.

## ⚖️ License

This project is released under the [Apache License 2.0](LICENSE).
