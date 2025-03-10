import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# File paths for LLM evaluation and METEOR scores
eval_file = "llm_evaluation.json"
meteor_file = "meteor_scores.json"

# Load data
def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_accuracy_and_meteor():
    """Analyze accuracy fluctuations and compare with METEOR scores."""
    eval_data = load_data(eval_file)
    meteor_data = load_data(meteor_file)

    # Convert to DataFrame
    df_eval = pd.DataFrame(eval_data)
    df_meteor = pd.DataFrame(meteor_data)

    # Ensure correct merging by matching "question" & "distractor_set"
    df = pd.merge(df_eval, df_meteor, on=["question", "passage", "correct_answer", "distractor_set"], suffixes=("_eval", "_meteor"))

    # Compute accuracy per distractor set
    accuracy_summary = df.groupby("distractor_set")["is_correct"].mean().reset_index()
    accuracy_summary["accuracy"] = accuracy_summary["is_correct"] * 100  # Convert to %

    # Compute separate METEOR score averages per distractor set
    meteor_summary = df.groupby("distractor_set")["meteor_scores"].apply(
        lambda x: np.mean([np.mean(list(scores.values())) for scores in x])
    ).reset_index()
    meteor_summary.rename(columns={"meteor_scores": "avg_meteor_score"}, inplace=True)

    # Merge accuracy and METEOR score summaries
    summary = pd.merge(accuracy_summary, meteor_summary, on="distractor_set")

    # Plot accuracy vs METEOR scores
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(summary["distractor_set"], summary["accuracy"], color='blue', alpha=0.6, label="Accuracy (%)")
    ax2.plot(summary["distractor_set"], summary["avg_meteor_score"], color='red', marker='o', label="Avg. METEOR Score")

    ax1.set_xlabel("Distractor Set")
    ax1.set_ylabel("Accuracy (%)", color='blue')
    ax2.set_ylabel("Avg. METEOR Score", color='red')
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 1)
    plt.title("LLM Accuracy vs METEOR Scores Across Distractor Sets")
    plt.legend()
    plt.show()

    print("\n📊 Corrected Accuracy & METEOR Summary:")
    print(summary.to_string(index=False))

# Run analysis
analyze_accuracy_and_meteor()