import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the evaluation results
input_file = "distractor_quality_results.json"
output_csv = "distractor_analysis_summary.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON data to DataFrame
df = pd.json_normalize(data)

# Drop unnecessary columns for visualization
df_melted = df.melt(
    id_vars=["question", "correct_answer", "distractor_set"],
    value_vars=[
        "meteor_scores",
        "semantic_similarities",
        "legal_relevance_scores",
        "grammar_scores",
        "distractor_diversity_score"
    ],
    var_name="metric",
    value_name="values"
)

# Explode list values (since each question has multiple distractors)
df_melted = df_melted.explode("values")

# Convert values to float
df_melted["values"] = df_melted["values"].astype(float)

# Compute summary statistics
summary_stats = df_melted.groupby(["distractor_set", "metric"])["values"].describe()

# Save to CSV
summary_stats.to_csv(output_csv)
print(f"✅ Summary statistics saved to {output_csv}")

# ---- VISUALIZATION ----

# Set seaborn style
sns.set(style="whitegrid")

# **1. Boxplot of all metrics across different distractor sets**
plt.figure(figsize=(12, 6))
sns.boxplot(x="metric", y="values", hue="distractor_set", data=df_melted)
plt.xticks(rotation=30)
plt.title("Distribution of Evaluation Metrics Across Distractor Sets")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig("boxplot_metrics.png")
print("✅ Boxplot saved as boxplot_metrics.png")

# **2. Heatmap of average metric scores**
pivot_table = df_melted.groupby(["distractor_set", "metric"])["values"].mean().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Average Metric Scores Per Distractor Set")
plt.savefig("heatmap_metrics.png")
print("✅ Heatmap saved as heatmap_metrics.png")

# **3. Scatter plot (Semantic Similarity vs. METEOR)**
plt.figure(figsize=(8, 5))

# Ensure proper filtering of only "semantic_similarities" and "meteor_scores"
scatter_data = df_melted[df_melted["metric"].isin(["semantic_similarities", "meteor_scores"])]

# Pivot to align correct x and y values for the scatterplot
scatter_data_pivot = scatter_data.pivot_table(index=["question", "distractor_set"], columns="metric", values="values").reset_index()

# Drop NaN rows (in case some questions lack values)
scatter_data_pivot = scatter_data_pivot.dropna()

# Generate scatterplot
sns.scatterplot(
    x=scatter_data_pivot["semantic_similarities"],
    y=scatter_data_pivot["meteor_scores"],
    hue=scatter_data_pivot["distractor_set"]
)
plt.xlabel("Semantic Similarity")
plt.ylabel("METEOR Score")
plt.title("Scatter Plot: Semantic Similarity vs. METEOR")
plt.savefig("scatter_meteor_vs_semantic.png")
print("✅ Scatter plot saved as scatter_meteor_vs_semantic.png")

print("📊 Data analysis complete!")