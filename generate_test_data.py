import json
import random

# Sample questions for various domains
SAMPLE_QUESTIONS = [
    # Law questions
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
        "question": "Under what circumstances can a private citizen legally use deadly force against another person?",
        "correct_answer": "When the person reasonably believes deadly force is necessary for self-defense against an imminent threat of death or serious bodily harm",
        "options": [
            "Whenever trespassing occurs on private property",
            "When the person reasonably believes deadly force is necessary for self-defense against an imminent threat of death or serious bodily harm",
            "Only when authorized by law enforcement officials",
            "Under no circumstances, as only law enforcement may use deadly force"
        ],
        "subject": "professional_law"
    },
    
    # Medicine questions
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
    },
    {
        "question": "Which of the following is the most common cause of community-acquired pneumonia?",
        "correct_answer": "Streptococcus pneumoniae",
        "options": [
            "Streptococcus pneumoniae",
            "Haemophilus influenzae",
            "Klebsiella pneumoniae",
            "Staphylococcus aureus"
        ],
        "subject": "professional_medicine"
    },
    
    # Mathematics questions
    {
        "question": "If f(x) = 3x² - 2x + 5, what is f'(x)?",
        "correct_answer": "6x - 2",
        "options": [
            "6x - 2",
            "3x² - 2",
            "6x² - 2",
            "6x - 2x"
        ],
        "subject": "high_school_mathematics"
    },
    {
        "question": "What is the solution to the system of equations: 2x + y = 5 and x - y = 1?",
        "correct_answer": "x = 2, y = 1",
        "options": [
            "x = 1, y = 3",
            "x = 3, y = -1",
            "x = 2, y = 1",
            "x = -1, y = 7"
        ],
        "subject": "high_school_mathematics"
    },
    
    # Computer Science questions
    {
        "question": "What is the time complexity of the quicksort algorithm in the average case?",
        "correct_answer": "O(n log n)",
        "options": [
            "O(n)",
            "O(n²)",
            "O(n log n)",
            "O(log n)"
        ],
        "subject": "college_computer_science"
    },
    {
        "question": "Which of the following is NOT a principle of object-oriented programming?",
        "correct_answer": "Procedural decomposition",
        "options": [
            "Inheritance",
            "Encapsulation",
            "Polymorphism",
            "Procedural decomposition"
        ],
        "subject": "college_computer_science"
    },
    
    # Physics questions
    {
        "question": "An object with mass m is moving with velocity v. What is its kinetic energy?",
        "correct_answer": "½mv²",
        "options": [
            "mv",
            "½mv²",
            "mv²",
            "m²v"
        ],
        "subject": "high_school_physics"
    },
    {
        "question": "Which of the following is Newton's Second Law of Motion?",
        "correct_answer": "F = ma",
        "options": [
            "For every action, there is an equal and opposite reaction",
            "An object in motion stays in motion unless acted upon by an external force",
            "F = ma",
            "Energy can neither be created nor destroyed"
        ],
        "subject": "high_school_physics"
    },
    
    # Biology questions
    {
        "question": "What is the primary function of mitochondria in a cell?",
        "correct_answer": "Energy production through cellular respiration",
        "options": [
            "Protein synthesis",
            "Energy production through cellular respiration",
            "Storage of genetic material",
            "Breakdown of waste products"
        ],
        "subject": "high_school_biology"
    },
    {
        "question": "Which of the following is NOT involved in photosynthesis?",
        "correct_answer": "Mitochondria",
        "options": [
            "Chloroplasts",
            "Carbon dioxide",
            "Mitochondria",
            "Sunlight"
        ],
        "subject": "high_school_biology"
    },
    
    # Philosophy questions
    {
        "question": "According to Kant's categorical imperative, one should:",
        "correct_answer": "Act only according to that maxim whereby you can, at the same time, will that it should become a universal law",
        "options": [
            "Always act to maximize pleasure and minimize pain",
            "Act only according to that maxim whereby you can, at the same time, will that it should become a universal law",
            "Act in accordance with virtue and moderation",
            "Always prioritize individual rights over collective welfare"
        ],
        "subject": "philosophy"
    },
    {
        "question": "Which philosopher is associated with the concept of the 'social contract'?",
        "correct_answer": "Jean-Jacques Rousseau",
        "options": [
            "Friedrich Nietzsche",
            "Aristotle",
            "Jean-Jacques Rousseau",
            "Søren Kierkegaard"
        ],
        "subject": "philosophy"
    },
    
    # History questions
    {
        "question": "The Treaty of Versailles was signed at the end of which war?",
        "correct_answer": "World War I",
        "options": [
            "World War I",
            "World War II",
            "The Franco-Prussian War",
            "The Thirty Years' War"
        ],
        "subject": "high_school_world_history"
    },
    {
        "question": "Who was the first Emperor of China?",
        "correct_answer": "Qin Shi Huang",
        "options": [
            "Qin Shi Huang",
            "Emperor Wu of Han",
            "Emperor Gaozu of Tang",
            "Kublai Khan"
        ],
        "subject": "high_school_world_history"
    },
    
    # Psychology questions
    {
        "question": "Which of the following is NOT one of Freud's psychosexual stages of development?",
        "correct_answer": "Cognitive stage",
        "options": [
            "Oral stage",
            "Phallic stage",
            "Latent stage",
            "Cognitive stage"
        ],
        "subject": "high_school_psychology"
    },
    {
        "question": "What is the recency effect in memory?",
        "correct_answer": "The tendency to better recall items that were presented last in a sequence",
        "options": [
            "The tendency to better recall items that were presented first in a sequence",
            "The tendency to better recall items that were presented last in a sequence",
            "The tendency to better recall traumatic memories",
            "The tendency to better recall recent events compared to events from long ago"
        ],
        "subject": "high_school_psychology"
    },
    
    # Nutrition questions
    {
        "question": "Which of the following is NOT a water-soluble vitamin?",
        "correct_answer": "Vitamin E",
        "options": [
            "Vitamin C",
            "Vitamin B12",
            "Vitamin E",
            "Vitamin B6"
        ],
        "subject": "nutrition"
    },
    {
        "question": "What is the primary role of fiber in the diet?",
        "correct_answer": "Promoting digestive health and regular bowel movements",
        "options": [
            "Providing energy",
            "Building muscle tissue",
            "Promoting digestive health and regular bowel movements",
            "Transporting oxygen in the blood"
        ],
        "subject": "nutrition"
    }
]

def generate_test_data(output_file="mmlu_expanded_dataset.json", num_questions=150):
    """Generate test data for the MMLU benchmark."""
    # Multiply the sample questions to reach the target number
    repeats_needed = max(1, num_questions // len(SAMPLE_QUESTIONS) + 1)
    expanded_questions = []
    
    for _ in range(repeats_needed):
        # Add some variation by shuffling
        shuffled_questions = SAMPLE_QUESTIONS.copy()
        random.shuffle(shuffled_questions)
        expanded_questions.extend(shuffled_questions)
    
    # Trim to the target number
    if len(expanded_questions) > num_questions:
        expanded_questions = expanded_questions[:num_questions]
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_questions, f, indent=4)
    
    print(f"✅ Generated {len(expanded_questions)} test questions")
    print(f"✅ Saved to {output_file}")
    return expanded_questions

if __name__ == "__main__":
    generate_test_data() 