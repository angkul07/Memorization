# Reducing Memorization Without Compromising Generalization in LLMs via Prompt-Driven Perturbation Training

> A lightweight, inference-driven approach to reduce memorization in large language models (LLMs) using GPT-4o-powered semantic perturbations and automatic filtering.

---

## Motivation

Modern large language models (LLMs) like GPT-4o demonstrate powerful reasoning abilities, especially in few-shot or in-context learning (ICL) settings. However, they remain vulnerable to **verbatim memorization** of training data—posing **privacy risks** and casting doubt on whether their outputs reflect true reasoning.

**This project proposes an inference-time data curation strategy that reduces memorization without compromising generalization**, leveraging the LLM ecosystem itself (GPT-4o + DeepEval/g-eval).

---

## Research Objective

We introduce **Perturbation-Aware Prompting (PAP)**—a novel training-time intervention that curates a **memorization-resilient training set** using LLM-generated semantic perturbations and filtering via g-eval. Our key hypotheses:

- **H1**: Models trained on perturbed data will exhibit **lower memorization**, measured using Adversarial Confidence Rating (ACR).
- **H2**: These models will **retain generalization** on unseen test samples, maintaining high reasoning accuracy.

---

## Key Contributions

- **Perturbation Framework**: Use GPT-4o to generate lexical, typographical, and structural perturbations of GSM8K answers.
- **Automatic Filtering with g-eval**: Retain only semantically valid perturbations using DeepEval's rule-based verification.
- **Memorization Assessment**: Apply ACR to detect memorization on training samples.
- **Generalization Benchmark**: Evaluate models on the GSM8K test set to ensure no accuracy drop.

---

## Methodology Overview

### 1. Dataset Preparation
- **Base**: [GSM8K dataset](https://huggingface.co/datasets/gsm8k)
- **Perturbations**:
  - Synonym swaps (lexical)
  - Minor typos (typographical)
  - Sentence reorderings (structural)

### 2. Perturbation Generation
- Prompts to GPT-4o create multiple perturbed answers per GSM8K sample.
- Stored and versioned under [`perturbation_generation_code/`](./datasets/perturbation_generation_code).

### 3. Filtering with g-eval
- Use DeepEval’s `g-eval` to verify:
  - Semantic consistency
  - Rule satisfaction
- Only pass-verified samples form the **GSM8K-P dataset** (`g-eval_filtered_perturbed_data/`).

### 4. Model Training & Evaluation
- **Variants**:
  - `Vanilla`: Trained on original GSM8K
  - `Perturbed`: Trained on GSM8K-P
- Evaluate using:
  - `ACR` (memorization)
  - GSM8K test accuracy (generalization)
 
> Every trained model is available [here](https://huggingface.co/sohamwasmatkar)

---

## Repository Structure

```
 project-root
├── datasets/
│   ├── g-eval_filtered_perturbed_data/      # g-eval filtered data which contains perturbed + original dataset mix
│   └── generalization_accuracy_data/        # Datasets contains accuracy labels dataset.
│
├── perturbation_generation_code/            # Jupyter notebooks for generating + filtering perturbations
│   ├── GSM8K_Perturbed.ipynb
│   ├── v1 dataset creation pipeline.ipynb
│   └── example perturbations.ipynb
│
├── eval_metric/                              # Code for mia-tuner and different evaluation metrics
│   └── acr_code/                             # Code for Adversarial Confidence Rating (ACR)
|       ├── acr-memorization/
│       ├── acr.py
│   ├── generalization_accuracy.ipynb
│   ├── mia_hybrid.py
│   └── memgen.py
│
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

| Metric              | Description                                                                | Goal        |
|---------------------|-----------------------------------------------------------------------------|-------------|
| **ACR (↓)**         | Model confidence on training set prompts (lower = less memorization)       | ↓ Lower     |
| **Test Accuracy (↑)** | Reasoning performance on GSM8K’s 1k test set (higher = better generalization) | ↑ Higher    |
| **g-eval Pass Rate (↑)** | % of perturbations passing semantic rule checks via g-eval             | ↑ Higher    |

---
