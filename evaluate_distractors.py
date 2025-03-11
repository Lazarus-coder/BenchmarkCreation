import json
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
import spacy
import string

# Ensure NLTK tokenizers are downloaded
nltk.download("punkt")

# Load NLP models
nlp = spacy.load("en_core_web_sm")  # Grammar check model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model

# Define input distractor files
distractor_files = [
    "mmlu_law_distractors_set1.json",
    "mmlu_law_distractors_set2.json",
    "mmlu_law_distractors_set3.json",
    "mmlu_law_distractors_set4.json",
    "mmlu_law_distractors_set5.json"
]

output_file = "distractor_quality_results.json"


def compute_meteor(reference, candidate):
    """Compute METEOR similarity score with tokenized input."""
    reference_tokens = word_tokenize(reference)  # Tokenize reference (correct answer)
    candidate_tokens = word_tokenize(candidate)  # Tokenize distractor
    return float(meteor_score([reference_tokens], candidate_tokens))  # Convert to float


def compute_semantic_similarity(correct_answer, distractors):
    """Compute cosine similarity between correct answer and distractors using embeddings."""
    correct_embedding = embedder.encode(correct_answer, convert_to_tensor=True)
    distractor_embeddings = embedder.encode(distractors, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(correct_embedding, distractor_embeddings)[0].cpu().numpy()
    return [float(s) for s in similarities.tolist()]  # Convert float32 to float


def compute_legal_relevance(distractors):
    """Check if distractors contain legal terms (simple keyword matching)."""
    legal_keywords = {"theft", "robbery", "assault", "battery", "fraud", "trespassing", "burglary", "law", "jurisdiction"}
    scores = []
    for d in distractors:
        tokens = set(d.lower().translate(str.maketrans("", "", string.punctuation)).split())
        scores.append(float(len(tokens.intersection(legal_keywords)) / len(tokens) if len(tokens) > 0 else 0))
    return scores


def compute_grammar_score(distractors):
    """Assess grammatical quality using Spacy (heuristic: number of dependency errors)."""
    scores = []
    for d in distractors:
        doc = nlp(d)
        errors = sum(1 for token in doc if token.dep_ == "dep")  # Count dependency errors
        scores.append(float(1 - (errors / len(doc) if len(doc) > 0 else 0)))  # Convert to float
    return scores


def compute_distractor_diversity(distractors):
    """Measure how different distractors are from each other (pairwise semantic distance)."""
    distractor_embeddings = embedder.encode(distractors, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(distractor_embeddings, distractor_embeddings).cpu().numpy()
    avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])  # Avoid diagonal values
    return float(1 - avg_similarity)  # Convert to float


def evaluate_distractors():
    """Evaluate multiple characteristics of the generated distractors."""
    results = []

    for file in distractor_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            question = entry["question"]
            correct_answer = entry["correct_answer"]
            distractors = list(entry["distractors"].values())

            # Compute evaluation metrics
            meteor_scores = [compute_meteor(correct_answer, d) for d in distractors]
            semantic_similarities = compute_semantic_similarity(correct_answer, distractors)
            legal_relevance = compute_legal_relevance(distractors)
            grammar_scores = compute_grammar_score(distractors)
            diversity_score = compute_distractor_diversity(distractors)

            # Store results
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "distractor_set": file.replace(".json", ""),
                "meteor_scores": meteor_scores,
                "semantic_similarities": semantic_similarities,
                "legal_relevance_scores": legal_relevance,
                "grammar_scores": grammar_scores,
                "distractor_diversity_score": diversity_score
            })

    # Convert everything to Python-native types before saving
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Distractor evaluation saved to {output_file}, ready for analysis.")


# Run the evaluation
evaluate_distractors()