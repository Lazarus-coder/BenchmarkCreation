import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load LLM Evaluation Results
with open("llm_evaluation_results.json", "r", encoding="utf-8") as f:
    llm_data = json.load(f)

# Convert accuracy and response time into a DataFrame
accuracy_df = pd.DataFrame.from_dict(llm_data["accuracy_summary"], orient="index", columns=["accuracy"])
response_time_df = pd.DataFrame.from_dict(llm_data["response_time_summary"], orient="index", columns=["response_time"])

# Merge accuracy and response time
df_llm = accuracy_df.join(response_time_df)
df_llm.reset_index(inplace=True)
df_llm.rename(columns={"index": "distractor_set"}, inplace=True)

# Load Distractor Quality Data
with open("distractor_quality_results.json", "r", encoding="utf-8") as f:
    quality_data = json.load(f)

# Convert distractor quality results to a DataFrame
quality_records = []
for entry in quality_data:
    record = {
        "distractor_set": entry["distractor_set"],
        "meteor_scores": sum(entry["meteor_scores"]) / len(entry["meteor_scores"]),
        "semantic_similarities": sum(entry["semantic_similarities"]) / len(entry["semantic_similarities"]),
        "legal_relevance_scores": sum(entry["legal_relevance_scores"]) / len(entry["legal_relevance_scores"]),
        "grammar_scores": sum(entry["grammar_scores"]) / len(entry["grammar_scores"]),
        "distractor_diversity_score": entry["distractor_diversity_score"]
    }
    quality_records.append(record)

quality_df = pd.DataFrame(quality_records)

# Merge LLM results with distractor quality metrics
df_combined = df_llm.merge(quality_df, on="distractor_set", how="left")

# Compute correlations
correlation_matrix = df_combined.drop(columns=["distractor_set"]).corr()
correlations = correlation_matrix["accuracy"].to_dict()  # Extract correlation with accuracy

# Identify highest & lowest correlations
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
top_correlation = sorted_correlations[0] if sorted_correlations else None
lowest_correlation = sorted_correlations[-1] if sorted_correlations else None

# Prepare summary
summary = {
    "average_accuracy": df_combined["accuracy"].mean(),
    "average_response_time": df_combined["response_time"].mean(),
    "top_correlation_with_accuracy": {"metric": top_correlation[0], "value": top_correlation[1]} if top_correlation else None,
    "lowest_correlation_with_accuracy": {"metric": lowest_correlation[0], "value": lowest_correlation[1]} if lowest_correlation else None,
    "full_correlation_matrix": correlations
}

# Save summary as JSON
with open("comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# **1. Scatter Plot: Semantic Similarity vs. Accuracy**
plt.figure(figsize=(8, 5))
sns.scatterplot(x="semantic_similarities", y="accuracy", hue="distractor_set", data=df_combined)
plt.xlabel("Avg Semantic Similarity")
plt.ylabel("Accuracy (%)")
plt.title("Impact of Semantic Similarity on LLM Accuracy")
plt.savefig("scatter_semantic_vs_accuracy.png")
print("✅ Scatter plot saved as scatter_semantic_vs_accuracy.png")

# **2. Scatter Plot: Distractor Diversity vs. Accuracy**
plt.figure(figsize=(8, 5))
sns.scatterplot(x="distractor_diversity_score", y="accuracy", hue="distractor_set", data=df_combined)
plt.xlabel("Distractor Diversity Score")
plt.ylabel("Accuracy (%)")
plt.title("Impact of Distractor Diversity on LLM Accuracy")
plt.savefig("scatter_diversity_vs_accuracy.png")
print("✅ Scatter plot saved as scatter_diversity_vs_accuracy.png")

# **3. Scatter Plot: Response Time vs. Semantic Similarity**
plt.figure(figsize=(8, 5))
sns.scatterplot(x="semantic_similarities", y="response_time", hue="distractor_set", data=df_combined)
plt.xlabel("Avg Semantic Similarity")
plt.ylabel("Avg Response Time (sec)")
plt.title("Does Higher Similarity Increase Response Time?")
plt.savefig("scatter_semantic_vs_response_time.png")
print("✅ Scatter plot saved as scatter_semantic_vs_response_time.png")

# **4. Heatmap of Correlations**
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between LLM Performance and Distractor Quality")
plt.savefig("heatmap_correlations.png")
print("✅ Heatmap saved as heatmap_correlations.png")

# **5. Bar Chart: Accuracy per Distractor Set**
plt.figure(figsize=(8, 5))
sns.barplot(x="distractor_set", y="accuracy", data=df_combined, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Distractor Set")
plt.ylabel("Accuracy (%)")
plt.title("LLM Accuracy Across Different Distractor Sets")
plt.savefig("barplot_accuracy.png")
print("✅ Bar chart saved as barplot_accuracy.png")

# **6. Box Plot: Response Time per Distractor Set**
plt.figure(figsize=(8, 5))
sns.boxplot(x="distractor_set", y="response_time", data=df_combined, palette="coolwarm")
plt.xticks(rotation=45)
plt.xlabel("Distractor Set")
plt.ylabel("Response Time (sec)")
plt.title("Response Time Distribution Across Distractor Sets")
plt.savefig("boxplot_response_time.png")
print("✅ Box plot saved as boxplot_response_time.png")

# **7. Line Plot: Accuracy & Response Time Trends**
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Accuracy
sns.lineplot(x="distractor_set", y="accuracy", data=df_combined, marker="o", label="Accuracy", ax=ax1, color="blue")
ax1.set_xlabel("Distractor Set")
ax1.set_ylabel("Accuracy (%)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Plot Response Time on Secondary Y-axis
ax2 = ax1.twinx()
sns.lineplot(x="distractor_set", y="response_time", data=df_combined, marker="s", label="Response Time", ax=ax2, color="red")
ax2.set_ylabel("Response Time (sec)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("LLM Accuracy & Response Time Trends")
fig.tight_layout()
plt.savefig("lineplot_accuracy_response_time.png")
print("✅ Line plot saved as lineplot_accuracy_response_time.png")

print("📊 Analysis complete! Check the visualizations and summary file.")