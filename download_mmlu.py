import os
import json
import requests
import random
from tqdm import tqdm

# Use a mix of different subjects from actual MMLU dataset
SUBJECTS = [
    "professional_law",               # Legal domain
    "professional_medicine",          # Medical domain
    "high_school_mathematics",        # Math domain
    "college_computer_science",       # Computer science
    "high_school_physics",            # Physics
    "high_school_biology",            # Biology
    "philosophy",                     # Philosophy
    "high_school_world_history",      # History
    "high_school_psychology",         # Psychology
    "nutrition"                       # Nutrition/health
]

# Create sample questions from mmlu_law_original.json as fallback
SAMPLE_QUESTIONS = [
    {
        "question": "In the 19th century, industrialization led to significant societal changes. Many thinkers debated whether economic progress justified the worsening conditions of the working class. Some argued that technological advancements ultimately benefited society, while others highlighted the exploitation of labor and worsening inequality. Based on this passage, what was the main ethical concern regarding industrialization?",
        "correct_answer": "The tension between economic progress and worker welfare",
        "options": [
            "The tension between economic progress and worker welfare",
            "The environmental impact of factories",
            "The decline of agricultural lifestyles",
            "The rapid pace of urban expansion"
        ],
        "subject": "philosophy"
    },
    {
        "question": "A plaintiff brings a products liability action against the manufacturer of a lawnmower. The plaintiff was injured while using the lawnmower when his foot slipped under the mower. The plaintiff's expert witness testifies that the blade guard was of a type that was more dangerous than alternative designs. The defendant's expert testifies that the blade guard was designed in accordance with government safety regulations. The jury should find for whom?",
        "correct_answer": "Either party, because compliance with government regulations does not preclude a finding that the product was defective.",
        "options": [
            "Defendant, because compliance with government regulations is a complete defense to a products liability action.",
            "Defendant, because compliance with government regulations creates a presumption that the product was not defective.",
            "Plaintiff, because the testimony of the plaintiff's expert should be given greater weight than the testimony of the defendant's expert.",
            "Either party, because compliance with government regulations does not preclude a finding that the product was defective."
        ],
        "subject": "professional_law"
    },
    {
        "question": "A patient presents with elevated blood pressure, headache, and dizziness. The most appropriate initial diagnostic test would be:",
        "correct_answer": "24-hour ambulatory blood pressure monitoring",
        "options": [
            "Brain MRI",
            "Echocardiogram",
            "24-hour ambulatory blood pressure monitoring",
            "Renal artery ultrasound"
        ],
        "subject": "professional_medicine"
    }
]

# URL template for MMLU datasets - using a direct link to raw files in the MMLU repo
BASE_URL = "https://raw.githubusercontent.com/hendrycks/test/master/mmlu"
OUTPUT_FILE = "mmlu_expanded_dataset.json"
TARGET_QUESTIONS = 150  # Target number of questions

def download_mmlu_subject(subject):
    """Download test data for a specific MMLU subject."""
    # Construct the proper URLs for the MMLU dataset
    test_url = f"{BASE_URL}/{subject}/test.csv"
    test_answer_url = f"{BASE_URL}/{subject}/test_answers.csv"
    
    print(f"Attempting to download {subject} dataset...")
    
    try:
        # Download test questions
        test_response = requests.get(test_url)
        test_response.raise_for_status()
        
        # Download test answers
        answer_response = requests.get(test_answer_url)
        answer_response.raise_for_status()
        
        # Parse questions and answers
        test_lines = test_response.text.strip().split('\n')
        answer_lines = answer_response.text.strip().split('\n')
        
        questions = []
        for i, (test_line, answer_line) in enumerate(zip(test_lines, answer_lines)):
            # Parse CSV content
            parts = test_line.split(',')
            if len(parts) < 5:  # Need at least question + 4 options
                continue
                
            question = parts[0].strip('"')
            options = [part.strip('"') for part in parts[1:]]
            
            # Get the correct answer (0=A, 1=B, 2=C, 3=D)
            try:
                answer_idx = int(answer_line.strip())
                correct_answer = options[answer_idx]
            except (ValueError, IndexError):
                continue
            
            # Format for our JSON structure
            entry = {
                "question": question,
                "correct_answer": correct_answer,
                "options": options,
                "subject": subject
            }
            
            questions.append(entry)
            
        print(f"  ✅ Downloaded {len(questions)} questions for {subject}")
        return questions
        
    except Exception as e:
        print(f"  ❌ Error downloading {subject}: {str(e)}")
        return []

def main():
    """Download MMLU questions from multiple subjects."""
    all_questions = []
    successful_subjects = 0
    
    for subject in tqdm(SUBJECTS, desc="Downloading subjects"):
        subject_questions = download_mmlu_subject(subject)
        if subject_questions:
            successful_subjects += 1
            all_questions.extend(subject_questions)
    
    # If we couldn't download any questions, use sample questions
    if len(all_questions) == 0:
        print("⚠️ Could not download any questions from MMLU repository.")
        print("Using built-in sample questions instead...")
        all_questions = SAMPLE_QUESTIONS * 50  # Replicate to get about 150 questions
    
    # If we have more than the target number, randomly sample
    if len(all_questions) > TARGET_QUESTIONS:
        all_questions = random.sample(all_questions, TARGET_QUESTIONS)
    
    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=4)
        
    print(f"✅ Prepared {len(all_questions)} questions from {successful_subjects} subjects")
    print(f"✅ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 