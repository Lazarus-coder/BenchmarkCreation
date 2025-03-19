import json
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor
import re

# Set API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is not set. Set it using 'export OPENAI_API_KEY=your_key_here'")

# Initialize OpenAI client without proxies parameter
client = OpenAI()

# Define OpenAI models to test
MODELS = [
    "gpt-4o",          # Most capable model
    "gpt-4o-mini",     # Smaller version
    "gpt-3.5-turbo",   # Older model for comparison
]

# Output directory
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_confidence(text):
    """Extract confidence score (0-100) from LLM response."""
    confidence_pattern = r"confidence:?\s*(\d+)"
    match = re.search(confidence_pattern, text.lower())
    
    if match:
        try:
            confidence = int(match.group(1))
            return min(100, max(0, confidence))  # Ensure 0-100 range
        except ValueError:
            pass
    
    # Default confidence level if not found
    return None

def extract_reasoning_steps(text):
    """Extract reasoning steps from the response."""
    lines = text.split('\n')
    reasoning_lines = []
    
    # Look for lines that might be part of the reasoning
    in_reasoning = False
    for line in lines:
        line = line.strip()
        
        # Check for reasoning section indicators
        if any(marker in line.lower() for marker in ["reasoning:", "step ", "thinking:", "first,", "second,", "third,"]):
            in_reasoning = True
            reasoning_lines.append(line)
        elif in_reasoning and line and not line.startswith("confidence:") and not line.startswith("answer:"):
            reasoning_lines.append(line)
        elif in_reasoning and (line.startswith("confidence:") or line.startswith("answer:") or "my answer is" in line.lower()):
            in_reasoning = False
    
    return "\n".join(reasoning_lines)

def analyze_reasoning(reasoning_text):
    """Analyze reasoning steps to get metrics about the thinking process."""
    if not reasoning_text:
        return {
            "reasoning_length": 0,
            "steps_count": 0,
            "uncertainty_markers": 0
        }
    
    # Count reasoning steps
    steps = len(re.findall(r"step \d|first[,:]|second[,:]|third[,:]|finally[,:]", reasoning_text.lower()))
    
    # Count uncertainty markers
    uncertainty_words = ["maybe", "perhaps", "possibly", "likely", "unlikely", "might", "could", "not sure", "unclear"]
    uncertainty_count = sum(reasoning_text.lower().count(word) for word in uncertainty_words)
    
    return {
        "reasoning_length": len(reasoning_text),
        "steps_count": max(1, steps),  # At least 1 step if there's any reasoning
        "uncertainty_markers": uncertainty_count
    }

def query_llm_with_confidence(model, question, options, request_reasoning=True):
    """Query LLM to answer MCQ with confidence estimation and reasoning."""
    prompt_parts = [
        f"Question: {question}\n",
        f"Options: {', '.join(options)}\n\n"
    ]
    
    if request_reasoning:
        prompt_parts.append(
            "Please think through this step-by-step before answering. "
            "First provide your reasoning, then clearly state your final answer.\n\n"
            "Please end your response with:\n"
            "Answer: [exact text of your chosen option]\n"
            "Confidence: [0-100] (your confidence in this answer)\n"
        )
    else:
        prompt_parts.append(
            "Respond with your answer choice exactly as it appears in the options.\n"
            "Also rate your confidence from 0-100, where 100 means absolutely certain.\n\n"
            "Format: Answer: [your answer]\nConfidence: [0-100]\n"
        )
    
    prompt = "".join(prompt_parts)
    
    try:
        start_time = time.time()  # Start timing

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering multiple-choice questions with explanation and confidence rating."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000 if request_reasoning else 150,
            temperature=0.2
        )

        end_time = time.time()  # End timing
        response_time = end_time - start_time
        response_text = response.choices[0].message.content.strip()
        
        # Extract answer from response
        answer_match = re.search(r"answer:?\s*(.+?)(?:\n|$|confidence)", response_text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Clean up the answer to match option text exactly
            for option in options:
                if option.lower() in answer.lower():
                    answer = option
                    break
        else:
            # Fallback: try to find an exact option in the response
            answer = next((opt for opt in options if opt in response_text), "Error: No clear answer")
        
        # Extract confidence
        confidence = extract_confidence(response_text)
        
        # Extract and analyze reasoning steps (if requested)
        reasoning_text = extract_reasoning_steps(response_text) if request_reasoning else ""
        reasoning_metrics = analyze_reasoning(reasoning_text)
        
        return {
            "answer": answer,
            "raw_response": response_text,
            "confidence": confidence,
            "response_time": response_time,
            "reasoning_text": reasoning_text,
            "reasoning_metrics": reasoning_metrics,
        }
        
    except Exception as e:
        print(f"⚠️ LLM API Error with model {model}: {e}")
        return {
            "answer": "Error",
            "raw_response": f"Error: {str(e)}",
            "confidence": None,
            "response_time": None,
            "reasoning_text": "",
            "reasoning_metrics": {"reasoning_length": 0, "steps_count": 0, "uncertainty_markers": 0}
        }

def process_question(args):
    """Process a single question for a specific model and distractor set."""
    model, entry, distractor_set_name, request_reasoning = args
    
    question = entry["question"]
    correct_answer = entry["correct_answer"]
    
    # Prepare options
    if "options" in entry:
        options = entry["options"][:]  # Copy original options
    else:
        options = list(entry["distractors"].values())[:]  # Copy distractors
    
    # Ensure correct answer is in the options
    if correct_answer not in options:
        options.append(correct_answer)
    
    # Shuffle options to prevent positional bias
    random.shuffle(options)
    
    # Query model
    result = query_llm_with_confidence(model, question, options, request_reasoning)
    
    # Check if answer is correct
    is_correct = result["answer"] == correct_answer
    
    return {
        "question": question,
        "subject": entry.get("subject", "unknown"),
        "correct_answer": correct_answer,
        "llm_answer": result["answer"],
        "is_correct": is_correct,
        "confidence": result["confidence"],
        "response_time": result["response_time"],
        "reasoning_text": result["reasoning_text"],
        "reasoning_metrics": result["reasoning_metrics"],
        "distractor_set": distractor_set_name,
        "distractor_type": entry.get("distractor_type", "original"),
        "model": model
    }

def evaluate_llms_on_dataset():
    """Evaluate multiple LLMs on different distractor sets with detailed metrics."""
    # Find all distractor set files
    distractor_files = glob.glob("distractor_sets/mmlu_distractors_*.json")
    original_dataset = "mmlu_expanded_dataset.json"
    
    if not os.path.exists(original_dataset):
        print(f"❌ Original dataset file {original_dataset} not found!")
        return
        
    if len(distractor_files) == 0:
        print("❌ No distractor sets found. Run MMLU-DG.py first!")
        return
    
    print(f"Found {len(distractor_files)} distractor sets + 1 original dataset")
    
    all_results = []
    
    # Process original dataset first
    with open(original_dataset, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # Limit to a reasonable number of questions for testing
    original_subset = original_data
    
    for model in MODELS:
        print(f"Evaluating model: {model} on original dataset...")
        
        # For detailed reasoning on a small subset
        detailed_subset = original_subset[:5]  # First 5 questions get detailed reasoning
        
        # Process questions with reasoning
        detailed_args = [(model, entry, "original", True) for entry in detailed_subset]
        with ThreadPoolExecutor(max_workers=3) as executor:
            detailed_results = list(tqdm(
                executor.map(process_question, detailed_args),
                total=len(detailed_args),
                desc=f"{model} detailed analysis"
            ))
        
        # Process remaining questions without detailed reasoning
        remaining_subset = original_subset[5:]
        remaining_args = [(model, entry, "original", False) for entry in remaining_subset]
        with ThreadPoolExecutor(max_workers=5) as executor:
            remaining_results = list(tqdm(
                executor.map(process_question, remaining_args),
                total=len(remaining_args),
                desc=f"{model} remaining questions"
            ))
            
        all_results.extend(detailed_results + remaining_results)
    
    # Process each distractor set
    for distractor_file in distractor_files:
        distractor_set_name = os.path.basename(distractor_file).replace(".json", "")
        
        with open(distractor_file, "r", encoding="utf-8") as f:
            distractor_data = json.load(f)
        
        for model in MODELS:
            print(f"Evaluating model: {model} on {distractor_set_name}...")
            
            # For detailed reasoning on a small subset
            detailed_subset = distractor_data[:5]  # First 5 questions
            
            # Process questions with reasoning
            detailed_args = [(model, entry, distractor_set_name, True) for entry in detailed_subset]
            with ThreadPoolExecutor(max_workers=3) as executor:
                detailed_results = list(tqdm(
                    executor.map(process_question, detailed_args),
                    total=len(detailed_args),
                    desc=f"{model} {distractor_set_name} detailed"
                ))
            
            # Process remaining questions without detailed reasoning
            remaining_subset = distractor_data[5:]
            remaining_args = [(model, entry, distractor_set_name, False) for entry in remaining_subset]
            with ThreadPoolExecutor(max_workers=5) as executor:
                remaining_results = list(tqdm(
                    executor.map(process_question, remaining_args),
                    total=len(remaining_args),
                    desc=f"{model} {distractor_set_name} remaining"
                ))
                
            all_results.extend(detailed_results + remaining_results)
    
    # Process results
    print("Processing results...")
    
    # Calculate summaries
    model_summaries = {}
    distractor_summaries = {}
    model_distractor_summaries = {}
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Calculate overall metrics
    for model in MODELS:
        model_data = df_results[df_results["model"] == model]
        
        if len(model_data) == 0:
            continue
            
        model_summaries[model] = {
            "accuracy": round(model_data["is_correct"].mean() * 100, 2),
            "avg_confidence": round(model_data["confidence"].mean(), 2),
            "avg_response_time": round(model_data["response_time"].mean(), 3),
            "confidence_calibration": round(
                1 - abs(model_data["is_correct"].mean() - model_data["confidence"].mean() / 100), 3
            ),
            "count": len(model_data)
        }
    
    # Calculate distractor type metrics
    distractor_types = df_results["distractor_set"].unique()
    for d_type in distractor_types:
        type_data = df_results[df_results["distractor_set"] == d_type]
        
        if len(type_data) == 0:
            continue
            
        distractor_summaries[d_type] = {
            "accuracy": round(type_data["is_correct"].mean() * 100, 2),
            "avg_confidence": round(type_data["confidence"].mean(), 2),
            "avg_response_time": round(type_data["response_time"].mean(), 3),
            "count": len(type_data)
        }
    
    # Calculate model × distractor metrics
    for model in MODELS:
        model_distractor_summaries[model] = {}
        
        for d_type in distractor_types:
            combo_data = df_results[(df_results["model"] == model) & (df_results["distractor_set"] == d_type)]
            
            if len(combo_data) == 0:
                continue
                
            model_distractor_summaries[model][d_type] = {
                "accuracy": round(combo_data["is_correct"].mean() * 100, 2),
                "avg_confidence": round(combo_data["confidence"].mean(), 2),
                "avg_response_time": round(combo_data["response_time"].mean(), 3),
                "count": len(combo_data)
            }
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "detailed_results": all_results,
            "model_summaries": model_summaries,
            "distractor_summaries": distractor_summaries,
            "model_distractor_summaries": model_distractor_summaries
        }, f, indent=4)
    
    print(f"✅ Enhanced evaluation results saved to {results_file}")
    
    # Create a CSV for easier analysis
    csv_file = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.csv")
    if not df_results.empty:
        # Drop the long text fields for the CSV
        df_csv = df_results.drop(columns=["reasoning_text", "reasoning_metrics"], errors="ignore")
        df_csv.to_csv(csv_file, index=False)
        print(f"✅ CSV data saved to {csv_file}")

    return all_results, model_summaries, distractor_summaries, model_distractor_summaries

if __name__ == "__main__":
    evaluate_llms_on_dataset() 