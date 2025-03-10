import random
from nltk.translate.meteor_score import meteor_score


def generate_distractors(question, correct_answer):
    """Simulate weak and moderate distractor generation for reading comprehension."""
    weak_distractors = [
        "The passage has no clear meaning",  # Absolute statement
        "Always, the author supports only one idea",  # Incorrect generalization
        "The text contradicts itself in every sentence"  # Extreme misinterpretation
    ]

    moderate_distractors = [
        "The author argues in favor of the topic, but with some contradictions",  # Partial truth + flaw
        "The passage mostly agrees with the premise, but ignores counterarguments",  # Plausible but not fully correct
        "While initially neutral, the author subtly dismisses opposing views"  # Slightly misleading interpretation
    ]

    return random.choice(weak_distractors), random.choice(moderate_distractors)


def evaluate_meteor(generated_distractor, expert_distractors):
    """Compute METEOR score by comparing against all expert distractors and taking the highest score."""
    return max(meteor_score([d.split()], generated_distractor.split()) for d in expert_distractors)


# More Complex MMLU MCQ (Reading Comprehension)
passage = (
    "In the 19th century, industrialization led to significant societal changes. "
    "Many thinkers debated whether economic progress justified the worsening conditions of the working class. "
    "Some argued that technological advancements ultimately benefited society, while others highlighted the exploitation "
    "of labor and worsening inequality. The author presents both perspectives but leans toward supporting gradual reform."
)
question = "What is the author's perspective on industrialization?"
correct_answer = "The author acknowledges both sides but ultimately supports gradual reform."
expert_distractors = [
    "The author fully endorses industrialization without critique.",
    "The author completely opposes economic progress.",
    "The passage is neutral and does not take a stance."
]

# Generate distractors
weak_distractor, moderate_distractor = generate_distractors(question, correct_answer)

# Evaluate distractors using METEOR (against all expert distractors)
weak_score = evaluate_meteor(weak_distractor, expert_distractors)
moderate_score = evaluate_meteor(moderate_distractor, expert_distractors)

# Display results
print(f"Passage: {passage}\n")
print(f"Question: {question}")
print(f"Correct Answer: {correct_answer}")
print(f"Expert Distractors: {expert_distractors}")
print(f"Generated Weak Distractor: {weak_distractor} (METEOR Score: {weak_score:.4f})")
print(f"Generated Moderate Distractor: {moderate_distractor} (METEOR Score: {moderate_score:.4f})")