import json
import os
from openai import OpenAI
import random

# Set API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set. Set it using 'export OPENAI_API_KEY=your_key_here'")

client = OpenAI(api_key=api_key)

# Include the original dataset in the evaluation
distractor_files = [
    "mmlu_law_original.json",  # Original dataset
    "mmlu_law_distractors_set1.json",
    "mmlu_law_distractors_set2.json",
    "mmlu_law_distractors_set3.json",
    "mmlu_law_distractors_set4.json",
    "mmlu_law_distractors_set5.json"
]

output_file = "llm_evaluation_results.json"


def query_llm(question, options):
    """Query LLM to answer the MCQ question."""
    prompt = (
        f"Read the question and select the best answer from the provided options.\n\n"
        f"Question: {question}\n\n"
        f"Options: {', '.join(options)}\n\n"
        f"Respond with the best answer choice exactly as it appears."
    )

    print(f"\n🔍 DEBUG: Sending to LLM:\n{prompt}")  # Debugging

    try:
        response = client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering multiple-choice questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM API Error: {e}")
        return "Error"


def evaluate_llm():
    """Evaluate LLM accuracy across the original and generated distractor sets."""
    results = []
    accuracy_summary = {}

    for file in distractor_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        correct_count = 0
        total_questions = len(data)

        for entry in data:
            question = entry["question"]
            correct_answer = entry["correct_answer"]

            # Handle original dataset differently (it stores full answer choices)
            if "options" in entry:
                options = entry["options"][:]  # Copy original options
            else:
                options = list(entry["distractors"].values())[:]  # Copy distractors

            # Ensure correct answer is in the options (fixing potential issue)
            if correct_answer not in options:
                options.append(correct_answer)

            # Shuffle options to prevent positional bias
            random.shuffle(options)

            # Debugging: Print question and options before sending to LLM
            print(f"\n📌 Testing question: {question}")
            print(f"✔️ Correct Answer: {correct_answer}")
            print(f"📊 Options Given: {options}")

            llm_answer = query_llm(question, options)
            is_correct = llm_answer == correct_answer

            if is_correct:
                correct_count += 1

            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "llm_answer": llm_answer,
                "is_correct": is_correct,
                "distractor_set": file.replace(".json", "")
            })

        # Calculate accuracy for this dataset
        accuracy = round((correct_count / total_questions) * 100, 2) if total_questions > 0 else 0.0
        accuracy_summary[file.replace(".json", "")] = accuracy

        print(f"✅ {file}: Accuracy = {accuracy}% ({correct_count}/{total_questions} correct)")

    # Save results in a structured format for analysis
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"detailed_results": results, "accuracy_summary": accuracy_summary}, f, indent=4)

    print(f"✅ LLM evaluation results saved to {output_file}, ready for analysis.")


# Run the evaluation
evaluate_llm()