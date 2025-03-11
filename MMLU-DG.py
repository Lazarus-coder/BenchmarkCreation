import json
import os
from openai import OpenAI

# Set OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set. Set it using 'export OPENAI_API_KEY=your_key_here'")

client = OpenAI(api_key=api_key)

original_output_file = "mmlu_law_original.json"

def generate_distractors(question, correct_answer, num_distractors):
    """Generate a set of distractors with the same count as the original."""
    prompt = (
        f"Given the following multiple-choice question and correct answer, generate {num_distractors} plausible but incorrect distractors.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Distractors:\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that generates plausible but incorrect multiple-choice distractors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150 + (20 * num_distractors),  # Adjust tokens based on needed output
            temperature=0.7
        )
        distractors = response.choices[0].message.content.strip().split("\n")
        return [d.strip() for d in distractors if d.strip()][:num_distractors]  # Ensure correct number of distractors
    except Exception as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return ["Failed to generate"] * num_distractors  # Fallback to ensure correct length

def generate_distractor_sets(output_files):
    """Generate 5 sets of distractors for MMLU Law questions, keeping the same number of distractors as the original."""
    with open(original_output_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    distractor_sets = [[] for _ in range(len(output_files))]

    for entry in original_data:
        question = entry["question"]
        correct_answer = entry["correct_answer"]
        num_distractors = len(entry["options"]) - 1  # Excluding correct answer

        print(f"\n✅ Processing Question: {question}")
        print(f"📌 Correct Answer: {correct_answer}")
        print(f"🔢 Expected Distractor Count: {num_distractors}")

        for i in range(len(output_files)):
            distractors = generate_distractors(question, correct_answer, num_distractors)

            entry_data = {
                "question": question,
                "correct_answer": correct_answer,
                "distractors": {str(idx + 1): distractors[idx] for idx in range(num_distractors)}
            }
            distractor_sets[i].append(entry_data)

    # Save each set to a separate file
    for i, output_file in enumerate(output_files):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(distractor_sets[i], f, indent=4)
        print(f"✅ Distractor set saved to {output_file}")

# Define output files for the 5 sets of distractors
output_files = [
    "mmlu_law_distractors_set1.json",
    "mmlu_law_distractors_set2.json",
    "mmlu_law_distractors_set3.json",
    "mmlu_law_distractors_set4.json",
    "mmlu_law_distractors_set5.json"
]

# Generate and save distractors
generate_distractor_sets(output_files)