# Evaluating the Effect of Distractor Quality in MCQs

This project studies how distractor quality in multiple-choice questions (MCQs) affects large language model (LLM) performance.  
Instead of treating distractors as generic incorrect options, we construct a controlled taxonomy of distractor types and evaluate how different kinds of misleading answer choices change model accuracy, calibration, and robustness.

## Overview

Multiple-choice benchmarks are widely used to evaluate language models, but benchmark difficulty is often treated as a property of the question itself rather than the quality of the distractors. This project focuses on that missing dimension.

We build a controlled evaluation pipeline on top of the MMLU benchmark and systematically compare how different distractor types influence model behavior across multiple academic and professional domains.

### Research Questions

- How can distractor quality in MCQs be measured more systematically?
- How do different distractor types affect LLM performance?
- Do harder distractors reveal weaknesses that standard benchmarks fail to capture?

## Key Contributions

- Constructed a balanced **150-question MCQ evaluation set** based on MMLU across diverse subject areas
- Introduced a controlled taxonomy of **5 distractor types**
- Built an evaluation workflow for comparing model performance under different distractor conditions
- Measured not only **accuracy**, but also **confidence calibration**, **response behavior**, and **psychometric properties**
- Applied **IRT**, **ANOVA**, and visualization-based analysis to identify model vulnerabilities

## Distractor Taxonomy

We generate five controlled distractor types:

1. **Semantic Confounders**  
   Distractors that are semantically close to the correct answer but subtly wrong

2. **Plausible Alternatives**  
   Factually incorrect answers that still appear reasonable

3. **Syntax Manipulations**  
   Options that mirror the surface structure of the correct answer while changing meaning

4. **Negation-Based Distractors**  
   Options that negate or contradict the correct answer

5. **Partial Truths**  
   Answers that mix correct information with incorrect assertions

## Dataset

The evaluation set is built from MMLU and includes questions from multiple domains such as:

- Professional Law
- Professional Medicine
- High School Mathematics
- College Computer Science
- High School Physics
- High School Biology
- Philosophy
- High School World History
- High School Psychology
- Nutrition

This design allows us to compare distractor effects across both professional and general academic knowledge settings.

## Evaluated Models

We evaluate the following OpenAI models:

- **GPT-4o**
- **GPT-4o-mini**
- **GPT-3.5-turbo**

Each model is tested on:

- the original MMLU-style questions
- five alternative distractor conditions generated using our taxonomy

## Evaluation Dimensions

This project goes beyond raw accuracy and includes multiple analysis layers:

- **Accuracy**
- **Confidence scores**
- **Response time**
- **Expected Calibration Error (ECE)**
- **Reasoning complexity**
- **Error typology**
- **Question difficulty and discrimination via IRT**
- **Statistical significance via ANOVA and post-hoc testing**

## Main Findings

### 1. Semantic confounders are the most damaging distractor type
Semantic confounders caused the largest overall performance drop across all evaluated models.

### 2. More capable models are still vulnerable
Although GPT-4o was more robust than GPT-4o-mini and GPT-3.5-turbo, even stronger models showed substantial degradation under controlled distractor conditions.

### 3. Partial truths are highly discriminative
Partial-truth distractors provided strong psychometric discrimination, making them especially useful for distinguishing model capability.

### 4. Confidence is often miscalibrated
Models frequently remained overconfident even when accuracy dropped, especially under more deceptive distractor conditions.

### 5. Benchmark performance is context-sensitive
The results suggest that MCQ benchmark scores are not fully stable indicators of “true knowledge”; they are significantly affected by distractor design.

## Project Workflow

The overall pipeline consists of the following stages:

1. **Question selection**
   - sample balanced MCQs from MMLU across multiple domains

2. **Controlled distractor generation**
   - generate distractors for each taxonomy type using GPT-4o

3. **Benchmark construction**
   - assemble structured evaluation sets in machine-readable format

4. **Model evaluation**
   - run multiple LLMs on each distractor condition
   - collect outputs, confidence, latency, and reasoning traces

5. **Analysis**
   - compute performance metrics
   - compare distractor effectiveness
   - run psychometric and statistical analysis
   - generate visualizations

## Repository Structure

'''text
.
├── data/                # benchmark data, generated distractor sets, intermediate files
├── prompts/             # prompt templates for distractor generation
├── experiments/         # evaluation scripts and experiment runners
├── analysis/            # statistical analysis, plots, calibration, IRT, ANOVA
├── outputs/             # CSV / JSON results and figures
├── paper/               # report or paper PDF
└── README.md

'''
Paper

Evaluating Effect of Distractor Quality in MCQs
Ao Jiang, Sheng Bi, Xinyi Liu

Why This Matters

Many LLM benchmarks assume that incorrect options are interchangeable, but this project shows that distractor design changes model performance in measurable and interpretable ways. Controlled distractor generation can therefore serve as a more sensitive tool for probing reasoning robustness, calibration, and domain-specific weakness.

Future Directions
	•	expand the dataset beyond 150 questions
	•	include non-OpenAI models for broader comparison
	•	design distractors targeting causal, temporal, or counterfactual reasoning
	•	improve automatic metrics for distractor quality
	•	optimize distractor generation beyond prompt-based methods

Citation

If you use this repository, please cite:
'''
@misc{jiang2025distractorquality,
  title={Evaluating the Effect of Distractor Quality in MCQs},
  author={Ao Jiang and Sheng Bi and Xinyi Liu},
  year={2025}
}
'''
