import json
import random
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# Load model and tokenizer
MODEL_NAME = "bert-base-uncased"  # Adjust as needed
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Load test data
def load_testsets():
    with open("testsets.json", "r") as f:
        return json.load(f)


def evaluate_model():
    data = load_testsets()
    results = []

    for item in data:
        question = item['question']
        correct_answer = item['correct_answer']

        for level, distractors in zip([
            "Strong", "Moderate", "Weak"],
                [item['strong_distractors'], [item['moderate_distractor']], [item['weak_distractor']]]):
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

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete. Results saved.")


# Run evaluation
evaluate_model()