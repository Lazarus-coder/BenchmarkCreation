# Enhanced MMLU Distractor Benchmark

An expanded benchmark for evaluating Large Language Models on multiple-choice questions with controlled distractors. This project systematically explores how different types of distractors affect LLM performance using psychometric analysis.

## Project Overview

This benchmark conducts a rigorous evaluation of how different types of distractors (incorrect answer choices) affect the performance of Large Language Models (LLMs) on multiple-choice questions. The project includes:

1. **Expanded Dataset**: 150 questions from diverse MMLU subjects including law, medicine, mathematics, computer science, physics, and more
2. **Controlled Distractor Generation**: Creates a taxonomy of distractor types with specific properties
3. **Multi-Model Evaluation**: Tests multiple LLM architectures with confidence scoring and reasoning analysis
4. **Advanced Analysis**: Applies psychometric analysis, statistical methods, and calibration assessment

## Key Features

- **Distractor Taxonomy**:
  - Semantic confounders: Similar meaning but factually incorrect
  - Plausible alternatives: Reasonable but wrong choices
  - Syntax manipulations: Similar structure but different meaning
  - Negation-based: Contradict the correct answer
  - Partial truths: Mix of correct and incorrect information

- **Enhanced Metrics**:
  - Confidence scoring and calibration analysis
  - Thinking time analysis with step-by-step reasoning
  - Psychometric analysis (using Item Response Theory)
  - Statistical significance testing with ANOVA

- **Comprehensive Visualization**:
  - Response time analysis
  - Confidence calibration diagrams
  - Distractor effectiveness heatmaps
  - Reasoning pattern analysis

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (set as environment variable)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_key_here
   ```

### Running the Benchmark

To run the complete pipeline:
```
python run_experiment.py
```

With specific stages:
```
python run_experiment.py --skip-download --skip-generation
```

## Pipeline Stages

1. **Dataset Preparation** (`download_mmlu.py`):
   - Downloads questions from multiple MMLU subjects
   - Creates a balanced dataset of 150 questions

2. **Distractor Generation** (`MMLU-DG.py`):
   - Generates 5 types of controlled distractors
   - Uses GPT-4o for high-quality distractor creation

3. **Model Evaluation** (`enhanced_test.py`):
   - Tests multiple OpenAI models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
   - Captures confidence scores and reasoning patterns

4. **Advanced Analysis** (`advanced_analysis.py`):
   - Performs psychometric analysis
   - Runs statistical tests (ANOVA)
   - Generates visualizations

## Results

Evaluation results are saved in the `evaluation_results/` directory, including:
- JSON data files with complete evaluation results
- CSV files with statistical analysis
- Visualizations in the `figures/` subdirectory

## Extending the Benchmark

- Add more LLM models by updating the `MODELS` list in `enhanced_test.py`
- Create new distractor types by expanding the `DISTRACTOR_TYPES` dictionary in `MMLU-DG.py`
- Add additional questions by modifying `download_mmlu.py`

## Future Work

- Human vs. LLM performance comparison
- More sophisticated reasoning analysis
- Distractor generation strategies that target specific weaknesses

## License

[MIT License](LICENSE)
