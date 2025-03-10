import json
import os
from openai import OpenAI
from datasets import load_dataset

# Set OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set. Set it using 'export OPENAI_API_KEY=your_key_here'")

client = OpenAI(api_key=api_key)

original_output_file = "race_original.json"

def save_original_race_data(num_samples=10):
    """Extracts original RACE data and saves it for reference."""
    dataset = load_dataset("race", "high")
    sample_data = dataset["validation"].select(range(num_samples))  # Subset for testing

    original_data = []

    for entry in sample_data:
        passage = entry.get("article", "")
        question = entry.get("question", "")
        options_list = entry.get("options", [])
        correct_answer = entry.get("answer", "")

        if not passage or not question or not options_list or not correct_answer:
            print("⚠️ Skipping entry due to missing fields.")
            continue

        # Convert correct answer letter (A, B, C, D) to actual text
        answer_index = ord(correct_answer) - ord('A')
        correct_answer_text = options_list[answer_index] if 0 <= answer_index < len(options_list) else "N/A"

        original_data.append({
            "passage": passage,
            "question": question,
            "correct_answer": correct_answer_text,
            "options": options_list
        })

    with open(original_output_file, "w", encoding="utf-8") as f:
        json.dump(original_data, f, indent=4)

    print(f"✅ Original RACE dataset saved to {original_output_file}")

# Run function to create race_original.json
save_original_race_data(num_samples=10)

def generate_distractors(question, correct_answer, difficulty):
    """
    Generate distractors using OpenAI's GPT-4o-mini.
    """
    prompt = (
        f"Given the following multiple-choice question and correct answer, generate three {difficulty} distractors that are plausible but incorrect.\n\n"
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
            max_tokens=100,
            temperature=0.7
        )
        distractors = response.choices[0].message.content.strip().split("\n")
        return [d.strip() for d in distractors if d.strip()][:3]  # Ensure only 3 distractors are returned
    except Exception as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return ["Failed to generate", "Failed to generate", "Failed to generate"]

def process_race_dataset(output_file_weak, output_file_moderate, num_samples=10):
    """
    Process the RACE dataset to generate weak and moderate distractors.
    """
    dataset = load_dataset("race", "high")
    sample_data = dataset["validation"].select(range(num_samples))  # Select subset for testing

    processed_data_weak = []
    processed_data_moderate = []

    for entry in sample_data:
        passage = entry.get("article", "")
        question = entry.get("question", "")  # Single question string
        options_list = entry.get("options", [])  # List of answer choices
        answer = entry.get("answer", "")  # Single correct answer letter

        if not passage or not question or not options_list or not answer:
            print("⚠️ Skipping entry due to missing fields.")
            continue

        try:
            # Convert 'A', 'B', 'C', 'D' to index (0,1,2,3)
            answer_index = ord(answer) - ord('A')
            correct_answer = options_list[answer_index] if 0 <= answer_index < len(options_list) else "N/A"

            print(f"\n✅ Processing Question: {question}")
            print(f"📌 Correct Answer: {correct_answer}")

            # Generate weak and moderate distractors
            weak_distractors = generate_distractors(question, correct_answer, "weak")
            moderate_distractors = generate_distractors(question, correct_answer, "moderate")

            # Ensure we have exactly 3 distractors; pad with placeholders if necessary
            while len(weak_distractors) < 3:
                weak_distractors.append("Generated failed")
            while len(moderate_distractors) < 3:
                moderate_distractors.append("Generated failed")

            entry_data_weak = {
                "passage": passage,
                "question": question,
                "correct_answer": correct_answer,
                "distractors": {
                    "1": weak_distractors[0],
                    "2": weak_distractors[1],
                    "3": weak_distractors[2]
                }
            }
            entry_data_moderate = {
                "passage": passage,
                "question": question,
                "correct_answer": correct_answer,
                "distractors": {
                    "1": moderate_distractors[0],
                    "2": moderate_distractors[1],
                    "3": moderate_distractors[2]
                }
            }

            processed_data_weak.append(entry_data_weak)
            processed_data_moderate.append(entry_data_moderate)

        except Exception as e:
            print(f"⚠️ Error processing question {question}: {e}")
            continue

    # Save the processed data to JSON files
    with open(output_file_weak, 'w', encoding='utf-8') as f:
        json.dump(processed_data_weak, f, indent=4)
    with open(output_file_moderate, 'w', encoding='utf-8') as f:
        json.dump(processed_data_moderate, f, indent=4)

    print(f"✅ Weak distractors dataset saved to {output_file_weak}")
    print(f"✅ Moderate distractors dataset saved to {output_file_moderate}")

# Example usage
process_race_dataset("race_weak_distractors.json", "race_moderate_distractors.json", num_samples=10)