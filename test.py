import json
import os
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openai import OpenAI

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
    """Query LLM to answer the MCQ question and measure response time."""
    prompt = (
        f"Read the question and select the best answer from the provided options.\n\n"
        f"Question: {question}\n\n"
        f"Options: {', '.join(options)}\n\n"
        f"Respond with the best answer choice exactly as it appears."
    )

    try:
        start_time = time.time()  # Start timing

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering multiple-choice questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )

        end_time = time.time()  # End timing
        response_time = end_time - start_time

        return response.choices[0].message.content.strip(), response_time
    except Exception as e:
        print(f"⚠️ LLM API Error: {e}")
        return "Error", None


def evaluate_llm():
    """Evaluate LLM accuracy and response time across different distractor sets."""
    results = []
    accuracy_summary = {}
    response_time_summary = {}

    for file in distractor_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        correct_count = 0
        total_questions = len(data)
        total_response_time = 0
        num_valid_responses = 0

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

            llm_answer, response_time = query_llm(question, options)
            is_correct = llm_answer == correct_answer

            if is_correct:
                correct_count += 1

            if response_time is not None:
                total_response_time += response_time
                num_valid_responses += 1

            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "llm_answer": llm_answer,
                "is_correct": is_correct,
                "response_time": response_time,
                "distractor_set": file.replace(".json", "")
            })

        # Calculate accuracy and average response time
        accuracy = round((correct_count / total_questions) * 100, 2) if total_questions > 0 else 0.0
        avg_response_time = round((total_response_time / num_valid_responses), 3) if num_valid_responses > 0 else None

        accuracy_summary[file.replace(".json", "")] = accuracy
        response_time_summary[file.replace(".json", "")] = avg_response_time

        print(f"✅ {file}: Accuracy = {accuracy}% ({correct_count}/{total_questions} correct), Avg Response Time = {avg_response_time} sec")

    # Save results in a structured format for analysis
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "detailed_results": results,
            "accuracy_summary": accuracy_summary,
            "response_time_summary": response_time_summary
        }, f, indent=4)

    print(f"✅ LLM evaluation results saved to {output_file}, ready for analysis.")


# Run the evaluation
evaluate_llm()