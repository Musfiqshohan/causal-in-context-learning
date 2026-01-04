

This repository is **to play with** the official implementation of **"CCL: Causal-aware In-context Learning for Out-of-Distribution Generalization"**. 

# CCL: Causal-aware In-context Learning for Out-of-Distribution Generalization

In this study, we focus on constructing a robust demonstration set to enhance the generalization of LLMs in OOD scenarios. Inspired by CRL, we propose a novel demonstration selection method, causal-aware in-context learning (CCL), which learns causal representations that remain invariant across environments and prioritizes candidates by assigning higher ranks to those with causal representations similar to the target query. Under the causal mechanism, we theoretically demonstrate that the demonstration set selected by CCL comprises candidates that are more closely related to the underlying problem addressed by the target query, rather than merely matching its context. The problem-level invariance of CCL ensures generalization performance for the target query even in unseen environments. We empirically validate that CCL operates robustly in OOD scenarios and demonstrates superior generalization performance on both synthetic and real datasets.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Dataset
[MGSM dataset](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mgsm)
[OOD NLP dataset](https://github.com/microsoft/LMOps/tree/main/llm_retriever) (the same datasets used for training and evaluating LLM-R).

## Training CCL

To train CCL, please refer [here](./ccl/README.md) to get detailed commond scripts

## Generating input prompt \& evaluating with LMs

To conduct few-shot or zero-shot prompt generation, please refer [here](./icl/README.md) to get detailed commond scripts

## Results

### MGSM

| Method     | Total |   ID  |  OOD  |
|------------|:-----:|:-----:|:-----:|
| ZS         | 87.71 | 89.43 | 84.70 |
| ICL (Fix.) | 91.20 | 91.26 | 91.10 |
| ICL (KNN)  | 94.07 | 95.83 | 91.00 |
| **CCL**    | **94.55** | **96.11** | **91.80** |

### OOD NLP


| Language Model   | Retrieval Method  | QNLI  | PIQA  | WSC273 | YELP  | Avg.  |
|------------------|-------------------|:-----:|:-----:|:------:|:-----:|:-----:|
| Llama 3.2 3B IT  | ZS                | 43.36 | **71.33** | 55.31  | 88.98 | 64.75 |
|                  | LLM-R             | 29.93 | 69.91 | 61.17  | 79.48 | 60.12 |
|                  | ICL (K-means)     | 68.13 | 69.04 | 49.82  | 75.81 | 65.70 |
|                  | **CCL**           | **75.18** | 70.46 | **61.91** | **95.44** | **75.74** |
|                  |                   |           |       |           |           |           |
| Phi-4 mini IT    | ZS                | **86.34** | **76.01** | 64.10  | 95.76 | 80.55 |
|                  | LLM-R             | 85.21 | 74.10 | 65.93  | 96.37 | 80.40 |
|                  | ICL (K-means)     | 83.18 | 74.81 | 71.06  | 96.25 | **81.33** |
|                  | **CCL**           | 82.26 | 75.73 | **71.43** | **96.33** | 81.44 |
|                  |                   |           |       |           |           |           |
| GPT-4o           | ZS                | **91.30** | 94.07 | 90.84  | 97.47 | 93.42 |
|                  | LLM-R             | 90.32 | **94.23** | 92.67  | 98.27 | 93.87 |
|                  | ICL (K-means)     | **88.28** | 93.04 | 87.55  | 98.17 | 91.76 |
|                  | **CCL**           | 90.77 | 93.15 | **93.77** | **98.36** | **94.01** |


Sentence embedding model: OpenAI’s text-embedding-3-small

## Base code of this repository.
The CCL code is based on [this repository](https://github.com/changliu00/causal-semantic-generative-model).
