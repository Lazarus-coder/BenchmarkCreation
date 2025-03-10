import json
import os
from openai import OpenAI

# Set API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set. Set it using 'export OPENAI_API_KEY=your_key_here'")

client = OpenAI(api_key=api_key)

distractor_files = ["race_weak_distractors.json", "race_moderate_distractors.json"]
output_file = "llm_evaluation.json"


def query_llm(passage, question, options):
    """Query LLM to answer the MCQ question."""
    prompt = (
        f"Read the passage and answer the question by selecting the best option.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n"
        f"Options: {', '.join(options)}\n\n"
        f"Respond with the best answer choice exactly as it appears."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    """Evaluate LLM accuracy with different distractor sets."""
    results = []

    for file in distractor_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            passage = entry["passage"]
            question = entry["question"]
            correct_answer = entry["correct_answer"]
            options = [correct_answer] + list(entry["distractors"].values())

            llm_answer = query_llm(passage, question, options)
            is_correct = llm_answer == correct_answer

            results.append({
                "passage": passage,
                "question": question,
                "correct_answer": correct_answer,
                "llm_answer": llm_answer,
                "is_correct": is_correct,
                "distractor_set": file.replace(".json", "")
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"✅ LLM evaluation saved to {output_file}, ready for analysis.")


# Example usage
evaluate_llm()
