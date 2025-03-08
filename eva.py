import openai
import random
import json
from transformers import AutoModelForMultipleChoice, AutoTokenizer, pipeline


# Load MMLU dataset (Placeholder function)
def load_mmlu_data():
    """Load original MCQs from the MMLU dataset."""
    with open("mmlu_data.json", "r") as f:
        return json.load(f)


# Generate distractors using LLM
def generate_distractor(prompt):
    """Use an LLM to generate distractors based on a given prompt."""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Adjust model accordingly
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response["choices"][0]["message"]["content"].strip()


# Generate weak and moderate distractors
def generate_distractors(question, correct_answer):
    """Generate weak and moderate distractors using LLM prompts."""
    weak_prompt = f"""
    Given the following question and its correct answer, generate a distractor that is obviously incorrect.
    The distractor should:
    - Include overt absolute terms (e.g., 'always,' 'never')
    - Be noticeably shorter than the correct answer
    - Exhibit minimal semantic overlap with the question context
    - Not require any deep reasoning to dismiss.
    Question: {question}
    Correct Answer: {correct_answer}
    """
    weak_distractor = generate_distractor(weak_prompt)

    moderate_prompt = f"""
    Given the following question and its correct answer, generate a distractor that appears plausible but includes subtle logical inconsistencies.
    The distractor should:
    - Be of similar length to the correct answer
    - Maintain moderate semantic relevance to the question
    - Contain minor structural or logical cues that hint at its inappropriateness
    - Require some reasoning to eliminate, but not as challenging as the expert-crafted version.
    Question: {question}
    Correct Answer: {correct_answer}
    """
    moderate_distractor = generate_distractor(moderate_prompt)

    return weak_distractor, moderate_distractor


# Evaluate model performance
def evaluate_model(model, tokenizer, data):
    """Evaluate model performance on different distractor levels."""
    results = []
    for item in data:
        question = item['question']
        correct_answer = item['correct_answer']
        strong_distractors = item['distractors']  # From MMLU dataset
        weak_distractor, moderate_distractor = generate_distractors(question, correct_answer)

        for level, distractors in zip([
            "Strong", "Moderate", "Weak"],
                [strong_distractors, [moderate_distractor], [weak_distractor]]):
            choices = [correct_answer] + distractors
            random.shuffle(choices)

            inputs = tokenizer([question] * len(choices), choices, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits.detach().numpy()
            pred_index = logits.argmax()
            is_correct = choices[pred_index] == correct_answer
            confidence = logits.max()

            results.append({
                "question": question,
                "level": level,
                "prediction": choices[pred_index],
                "correct": is_correct,
                "confidence": confidence
            })

    return results


# Load model and tokenizer
model_name = "bert-base-uncased"  # Replace with your evaluation model
model = AutoModelForMultipleChoice.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load data and run evaluation
data = load_mmlu_data()
results = evaluate_model(model, tokenizer, data)

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation complete. Results saved.")
