import json
import os
from nltk.translate.meteor_score import meteor_score

# File paths for input and output
distractor_files = ["race_weak_distractors.json", "race_moderate_distractors.json"]
original_data_file = "race_original.json"  # Stores original options for reference
output_file = "meteor_scores.json"


def clean_text(text):
    """Normalize text by lowercasing and tokenizing for METEOR."""
    return text.lower().split()  # Ensure it returns a tokenized list


def compute_meteor(reference, hypothesis):
    """Compute METEOR score between reference and hypothesis."""
    return meteor_score([clean_text(reference)], clean_text(hypothesis))


def load_original_data():
    """Load original answer options for accurate METEOR scoring."""
    if not os.path.exists(original_data_file):
        raise FileNotFoundError(f"❌ Original data file '{original_data_file}' not found!")

    with open(original_data_file, 'r', encoding='utf-8') as f:
        return {entry["question"]: entry for entry in json.load(f)}


def process_distractors():
    """Load generated distractors, compute METEOR scores, and save results."""
    original_data = load_original_data()
    results = []

    for file in distractor_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            passage = entry["passage"]
            question = entry["question"]
            correct_answer = entry["correct_answer"]
            distractors = entry["distractors"]

            if question not in original_data:
                print(f"⚠️ Skipping question not found in original dataset: {question}")
                continue

            reference_answer = original_data[question]["correct_answer"]  # Get original correct answer

            meteor_scores = {
                "1": compute_meteor(reference_answer, distractors["1"]),
                "2": compute_meteor(reference_answer, distractors["2"]),
                "3": compute_meteor(reference_answer, distractors["3"])
            }

            results.append({
                "passage": passage,
                "question": question,
                "correct_answer": correct_answer,
                "reference_answer": reference_answer,
                "meteor_scores": meteor_scores,
                "distractor_set": file.replace(".json", "")
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"✅ METEOR scores saved to {output_file}, ready for analysis in data_analysis.py")


# Example usage
process_distractors()
