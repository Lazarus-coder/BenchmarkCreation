# MMLU Distractor Experiment: To-Do List & Roadmap

## To-Do List
1. **Extract MMLU Data**
   - Load original questions, answers, and expert-crafted distractors from the dataset.
2. **Generate Distractors Using DeepInfra API**
   - Implement API calls to generate weak and moderate distractors.
   - Ensure distractors align with predefined difficulty levels.
   - Store test sets locally.
3. **Evaluation Script**
   - Load different distractor sets (strong, moderate, weak).
   - Use a language model to evaluate performance.
   - Compute accuracy and confidence metrics.
   - Save evaluation results for analysis.
4. **Analysis & Reporting**
   - Compare model performance across distractor levels.
   - Identify patterns in model weaknesses or strengths.
   - Present results for further study.

## Roadmap
### Phase 1: Data Processing (Current Phase)
- [x] Load MMLU dataset.
- [ ] Integrate DeepInfra API for distractor generation.
- [ ] Save modified dataset with weak and moderate distractors.

### Phase 2: Evaluation Implementation
- [ ] Implement evaluation using an LLM model.
- [ ] Compute accuracy and confidence metrics.
- [ ] Save evaluation results.

### Phase 3: Analysis & Reporting
- [ ] Compare model behavior across different distractor sets.
- [ ] Identify trends and insights.
- [ ] Document findings.

---
