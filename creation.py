import requests
import json

API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
API_KEY = "your_api_key_here"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# Load MMLU dataset
def load_mmlu_data():
    with open("mmlu_data.json", "r") as f:
        return json.load(f)


# Generate distractors using DeepInfra API
def generate_distractor(prompt):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",  # Adjust model if needed
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": prompt}],
        "max_tokens": 50
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response_data = response.json()
    return response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


# Generate weak and moderate distractors
def generate_distractors(question, correct_answer):
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


# Process MMLU dataset
def process_mmlu_data():
    data = load_mmlu_data()
    testsets = []

    for item in data:
        question = item['question']
        correct_answer = item['correct_answer']
        strong_distractors = item['distractors']  # From MMLU dataset
        weak_distractor, moderate_distractor = generate_distractors(question, correct_answer)

        testsets.append({
            "question": question,
            "correct_answer": correct_answer,
            "strong_distractors": strong_distractors,
            "moderate_distractor": moderate_distractor,
            "weak_distractor": weak_distractor
        })

    with open("testsets.json", "w") as f:
        json.dump(testsets, f, indent=4)

    print("Test sets saved successfully.")


# Run extraction and generation process
process_mmlu_data()