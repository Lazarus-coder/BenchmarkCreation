import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import language_tool_python
from scipy.spatial.distance import pdist, squareform
import itertools

# Download nltk resources if needed
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

# Suppress warnings
warnings.filterwarnings('ignore')

# Output directories
OUTPUT_DIR = "evaluation_results"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "additional_metrics")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load the evaluation results
results_file = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.json")
csv_file = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.csv")

# Initialize NLP tools
try:
    # Initialize sentence transformer model for semantic similarity
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize language tool for grammar checking
    language_tool = language_tool_python.LanguageTool('en-US')
except Exception as e:
    print(f"⚠️ Warning: Could not initialize NLP tools: {e}")
    embedding_model = None
    language_tool = None

def load_results():
    """Load evaluation results from JSON and CSV files."""
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
            
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            # Convert detailed results to DataFrame
            df = pd.DataFrame(results_data["detailed_results"])
            
        return results_data, df
    except FileNotFoundError:
        print(f"❌ Results file not found at {results_file}")
        return None, None

def perform_psychometric_analysis(df):
    """
    Perform psychometric analysis of distractor effectiveness using Item Response Theory concepts.
    
    This is a simplified version of IRT that calculates:
    - Item difficulty (p-value)
    - Item discrimination
    - Distractor analysis
    """
    print("Performing psychometric analysis...")
    
    # Group by question and calculate metrics
    item_stats = []
    
    # Get unique questions and models
    questions = df["question"].unique()
    models = df["model"].unique()
    
    for question in questions:
        question_data = df[df["question"] == question]
        
        # Calculate difficulty (p-value) - proportion of correct responses
        p_value = question_data["is_correct"].mean()
        
        # Calculate discrimination - correlation between getting this item right
        # and overall score (using point-biserial correlation)
        overall_scores = {}
        for model_name in models:
            model_df = df[df["model"] == model_name]
            overall_scores[model_name] = model_df["is_correct"].mean()
            
        # Add overall score to each row
        question_data = question_data.copy()
        question_data["overall_score"] = question_data["model"].map(overall_scores)
        
        # Calculate point-biserial correlation
        if len(question_data) > 1 and question_data["is_correct"].nunique() > 1:
            discrimination = question_data["is_correct"].corr(question_data["overall_score"])
        else:
            discrimination = np.nan
            
        # Get confidence statistics
        avg_confidence = question_data["confidence"].mean()
        correct_confidence = question_data[question_data["is_correct"]]["confidence"].mean() if any(question_data["is_correct"]) else np.nan
        incorrect_confidence = question_data[~question_data["is_correct"]]["confidence"].mean() if any(~question_data["is_correct"]) else np.nan
        
        # Calculate confidence discrimination (difference between correct and incorrect confidence)
        confidence_discrimination = correct_confidence - incorrect_confidence if not (np.isnan(correct_confidence) or np.isnan(incorrect_confidence)) else np.nan
        
        # Calculate response time statistics
        avg_response_time = question_data["response_time"].mean()
        correct_time = question_data[question_data["is_correct"]]["response_time"].mean() if any(question_data["is_correct"]) else np.nan
        incorrect_time = question_data[~question_data["is_correct"]]["response_time"].mean() if any(~question_data["is_correct"]) else np.nan
        
        # Find the majority distractor chosen when wrong
        if any(~question_data["is_correct"]):
            wrong_answers = question_data[~question_data["is_correct"]]["llm_answer"].value_counts()
            most_chosen_wrong = wrong_answers.index[0] if len(wrong_answers) > 0 else "N/A"
            wrong_count = wrong_answers.iloc[0] if len(wrong_answers) > 0 else 0
        else:
            most_chosen_wrong = "N/A"
            wrong_count = 0
        
        # Get subject info 
        subject = question_data["subject"].iloc[0] if "subject" in question_data.columns else "unknown"
        
        # Store item statistics
        item_stats.append({
            "question": question,
            "subject": subject,
            "p_value": p_value,
            "difficulty": 1 - p_value,  # Convert p-value to difficulty
            "discrimination": discrimination,
            "avg_confidence": avg_confidence,
            "correct_confidence": correct_confidence,
            "incorrect_confidence": incorrect_confidence,
            "confidence_discrimination": confidence_discrimination,
            "avg_response_time": avg_response_time,
            "correct_time": correct_time,
            "incorrect_time": incorrect_time,
            "most_chosen_wrong": most_chosen_wrong,
            "wrong_count": wrong_count
        })
    
    # Convert to DataFrame
    item_df = pd.DataFrame(item_stats)
    
    # Save psychometric analysis to CSV
    psychometric_file = os.path.join(OUTPUT_DIR, "psychometric_analysis.csv")
    item_df.to_csv(psychometric_file, index=False)
    print(f"✅ Psychometric analysis saved to {psychometric_file}")
    
    # Create visualizations
    create_psychometric_plots(item_df)
    
    return item_df

def create_psychometric_plots(item_df):
    """Create visualizations for psychometric analysis."""
    # 1. Item Difficulty vs. Discrimination Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=item_df, 
        x="difficulty", 
        y="discrimination",
        hue="subject",
        alpha=0.7,
        s=80
    )
    plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    plt.title("Item Difficulty vs. Discrimination")
    plt.xlabel("Difficulty (1 - p-value)")
    plt.ylabel("Discrimination")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "difficulty_vs_discrimination.png"))
    plt.close()
    
    # 2. Response Time vs. Difficulty
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=item_df, 
        x="difficulty", 
        y="avg_response_time",
        hue="subject",
        alpha=0.7,
        s=80
    )
    plt.title("Item Difficulty vs. Response Time")
    plt.xlabel("Difficulty (1 - p-value)")
    plt.ylabel("Average Response Time (sec)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "difficulty_vs_response_time.png"))
    plt.close()
    
    # 3. Confidence vs. Difficulty
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=item_df, 
        x="difficulty", 
        y="avg_confidence",
        hue="subject",
        alpha=0.7,
        s=80
    )
    plt.title("Item Difficulty vs. Confidence")
    plt.xlabel("Difficulty (1 - p-value)")
    plt.ylabel("Average Confidence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "difficulty_vs_confidence.png"))
    plt.close()
    
    # 4. Confidence Discrimination Plot
    # Only use items with valid confidence discrimination
    valid_conf_df = item_df.dropna(subset=["confidence_discrimination"])
    if len(valid_conf_df) > 5:  # Only if we have enough data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=valid_conf_df, 
            x="discrimination", 
            y="confidence_discrimination",
            hue="subject",
            alpha=0.7,
            s=80
        )
        plt.title("Item Discrimination vs. Confidence Discrimination")
        plt.xlabel("Item Discrimination")
        plt.ylabel("Confidence Discrimination\n(Correct - Incorrect Confidence)")
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "discrimination_vs_confidence.png"))
        plt.close()

def analyze_by_distractor_type(df):
    """
    Advanced analysis to understand how different distractor types affect LLM performance
    across different models.
    """
    print("Analyzing performance by distractor type...")
    
    # Calculate statistics by distractor_set
    distractor_stats = df.groupby(["model", "distractor_set"]).agg({
        "is_correct": ["mean", "count"],
        "confidence": ["mean", "std"],
        "response_time": ["mean", "std"]
    }).reset_index()
    
    # Flatten the multi-index columns
    distractor_stats.columns = ["_".join(col).strip("_") for col in distractor_stats.columns.values]
    
    # Calculate confidence calibration
    distractor_stats["calibration_error"] = abs(distractor_stats["is_correct_mean"] - distractor_stats["confidence_mean"]/100)
    
    # Save to CSV
    distractor_file = os.path.join(OUTPUT_DIR, "distractor_type_analysis.csv")
    distractor_stats.to_csv(distractor_file, index=False)
    print(f"✅ Distractor type analysis saved to {distractor_file}")
    
    # Create visualizations
    create_distractor_type_plots(distractor_stats, df)
    
    # Perform ANOVA to test for significant differences
    perform_anova_analysis(df)
    
    return distractor_stats

def create_distractor_type_plots(distractor_stats, full_df):
    """Create visualizations for distractor type analysis."""
    # 1. Accuracy by Model and Distractor Type
    plt.figure(figsize=(12, 6))
    accuracy_df = distractor_stats.pivot(
        index="distractor_set", 
        columns="model", 
        values="is_correct_mean"
    )
    sns.heatmap(
        accuracy_df, 
        annot=True, 
        cmap="YlGnBu", 
        fmt=".3f",
        vmin=0, 
        vmax=1,
        linewidths=0.5
    )
    plt.title("Accuracy by Model and Distractor Type")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "accuracy_by_model_distractor.png"))
    plt.close()
    
    # 2. Confidence by Model and Distractor Type
    plt.figure(figsize=(12, 6))
    confidence_df = distractor_stats.pivot(
        index="distractor_set", 
        columns="model", 
        values="confidence_mean"
    )
    sns.heatmap(
        confidence_df, 
        annot=True, 
        cmap="YlOrRd", 
        fmt=".1f",
        linewidths=0.5
    )
    plt.title("Average Confidence by Model and Distractor Type")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "confidence_by_model_distractor.png"))
    plt.close()
    
    # 3. Response Time by Model and Distractor Type
    plt.figure(figsize=(12, 6))
    time_df = distractor_stats.pivot(
        index="distractor_set", 
        columns="model", 
        values="response_time_mean"
    )
    sns.heatmap(
        time_df, 
        annot=True, 
        cmap="Greens", 
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Average Response Time by Model and Distractor Type")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "response_time_by_model_distractor.png"))
    plt.close()
    
    # 4. Calibration Error by Model and Distractor Type
    plt.figure(figsize=(12, 6))
    calibration_df = distractor_stats.pivot(
        index="distractor_set", 
        columns="model", 
        values="calibration_error"
    )
    sns.heatmap(
        calibration_df, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".3f",
        linewidths=0.5
    )
    plt.title("Calibration Error by Model and Distractor Type\n(Lower is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "calibration_by_model_distractor.png"))
    plt.close()
    
    # 5. Box plots comparing accuracy across distractor types
    plt.figure(figsize=(14, 6))
    sns.boxplot(x="distractor_set", y="is_correct", hue="model", data=full_df)
    plt.title("Accuracy Distribution by Distractor Type and Model")
    plt.xlabel("Distractor Type")
    plt.ylabel("Accuracy (1=Correct, 0=Incorrect)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "accuracy_boxplot_by_distractor.png"))
    plt.close()
    
    # 6. Box plots comparing confidence across distractor types
    plt.figure(figsize=(14, 6))
    sns.boxplot(x="distractor_set", y="confidence", hue="model", data=full_df)
    plt.title("Confidence Distribution by Distractor Type and Model")
    plt.xlabel("Distractor Type")
    plt.ylabel("Confidence Score (0-100)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "confidence_boxplot_by_distractor.png"))
    plt.close()

def perform_anova_analysis(df):
    """
    Perform ANOVA and post-hoc tests to determine if differences 
    between distractor types are statistically significant.
    """
    print("Performing ANOVA analysis...")
    
    # Create ANOVA results directory
    anova_dir = os.path.join(OUTPUT_DIR, "anova_results")
    os.makedirs(anova_dir, exist_ok=True)
    
    # Convert boolean is_correct to int for analysis
    df["correct_int"] = df["is_correct"].astype(int)
    
    # Perform ANOVA for accuracy ~ distractor_type for each model
    models = df["model"].unique()
    
    anova_results = []
    posthoc_results = []
    
    for model in models:
        model_df = df[df["model"] == model]
        
        # 1. ANOVA for accuracy ~ distractor_set
        try:
            accuracy_anova = pg.anova(
                data=model_df,
                dv="correct_int",
                between="distractor_set",
                detailed=True
            )
            
            accuracy_anova["model"] = model
            accuracy_anova["dependent_var"] = "accuracy"
            anova_results.append(accuracy_anova)
            
            # Post-hoc tests if ANOVA is significant
            if accuracy_anova["p-unc"].iloc[0] < 0.05:
                accuracy_posthoc = pg.pairwise_tukey(
                    data=model_df,
                    dv="correct_int",
                    between="distractor_set"
                )
                accuracy_posthoc["model"] = model
                accuracy_posthoc["dependent_var"] = "accuracy"
                posthoc_results.append(accuracy_posthoc)
        except Exception as e:
            print(f"  ⚠️ ANOVA error for accuracy with model {model}: {str(e)}")
        
        # 2. ANOVA for confidence ~ distractor_set
        try:
            confidence_anova = pg.anova(
                data=model_df,
                dv="confidence",
                between="distractor_set",
                detailed=True
            )
            
            confidence_anova["model"] = model
            confidence_anova["dependent_var"] = "confidence"
            anova_results.append(confidence_anova)
            
            # Post-hoc tests if ANOVA is significant
            if confidence_anova["p-unc"].iloc[0] < 0.05:
                confidence_posthoc = pg.pairwise_tukey(
                    data=model_df,
                    dv="confidence",
                    between="distractor_set"
                )
                confidence_posthoc["model"] = model
                confidence_posthoc["dependent_var"] = "confidence"
                posthoc_results.append(confidence_posthoc)
        except Exception as e:
            print(f"  ⚠️ ANOVA error for confidence with model {model}: {str(e)}")
        
        # 3. ANOVA for response_time ~ distractor_set
        try:
            time_anova = pg.anova(
                data=model_df,
                dv="response_time",
                between="distractor_set",
                detailed=True
            )
            
            time_anova["model"] = model
            time_anova["dependent_var"] = "response_time"
            anova_results.append(time_anova)
            
            # Post-hoc tests if ANOVA is significant
            if time_anova["p-unc"].iloc[0] < 0.05:
                time_posthoc = pg.pairwise_tukey(
                    data=model_df,
                    dv="response_time",
                    between="distractor_set"
                )
                time_posthoc["model"] = model
                time_posthoc["dependent_var"] = "response_time"
                posthoc_results.append(time_posthoc)
        except Exception as e:
            print(f"  ⚠️ ANOVA error for response_time with model {model}: {str(e)}")
    
    # Combine all ANOVA results
    if anova_results:
        all_anova = pd.concat(anova_results, ignore_index=True)
        anova_file = os.path.join(anova_dir, "anova_results.csv")
        all_anova.to_csv(anova_file, index=False)
        print(f"  ✅ ANOVA results saved to {anova_file}")
    
    # Combine all post-hoc results
    if posthoc_results:
        all_posthoc = pd.concat(posthoc_results, ignore_index=True)
        posthoc_file = os.path.join(anova_dir, "posthoc_results.csv")
        all_posthoc.to_csv(posthoc_file, index=False)
        print(f"  ✅ Post-hoc results saved to {posthoc_file}")
    
    return anova_results, posthoc_results

def analyze_reasoning_patterns(df):
    """
    Analyze reasoning patterns in LLM responses to categorize error types
    and identify patterns in thinking processes.
    """
    print("Analyzing reasoning patterns...")
    
    # Check if reasoning_text column exists
    if "reasoning_text" not in df.columns:
        print("  ⚠️ No reasoning_text column found in data - skipping reasoning analysis")
        return None
    
    # Filter for entries with reasoning text
    reasoning_df = df[df["reasoning_text"].str.len() > 0].copy()
    
    if len(reasoning_df) == 0:
        print("  ⚠️ No reasoning data available for analysis")
        return None
    
    # Extract reasoning metrics from DataFrame
    reasoning_metrics = []
    
    for _, row in reasoning_df.iterrows():
        # Extract metrics from the reasoning_metrics dictionary column
        if isinstance(row.get("reasoning_metrics"), dict):
            metrics = row["reasoning_metrics"]
        else:
            # Try to parse from string if it's not already a dict
            try:
                metrics = json.loads(row["reasoning_metrics"].replace("'", "\""))
            except:
                metrics = {"reasoning_length": 0, "steps_count": 0, "uncertainty_markers": 0}
        
        metrics_entry = {
            "model": row["model"],
            "distractor_set": row["distractor_set"],
            "is_correct": row["is_correct"],
            "confidence": row["confidence"],
            "response_time": row["response_time"],
            "reasoning_length": metrics.get("reasoning_length", 0),
            "steps_count": metrics.get("steps_count", 0),
            "uncertainty_markers": metrics.get("uncertainty_markers", 0)
        }
        
        reasoning_metrics.append(metrics_entry)
    
    # Convert to DataFrame
    reasoning_metrics_df = pd.DataFrame(reasoning_metrics)
    
    # Save to CSV
    reasoning_file = os.path.join(OUTPUT_DIR, "reasoning_analysis.csv")
    reasoning_metrics_df.to_csv(reasoning_file, index=False)
    print(f"✅ Reasoning analysis saved to {reasoning_file}")
    
    # Create visualizations
    create_reasoning_plots(reasoning_metrics_df)
    
    return reasoning_metrics_df

def create_reasoning_plots(reasoning_df):
    """Create visualizations for reasoning pattern analysis."""
    if len(reasoning_df) == 0:
        return
    
    # 1. Reasoning Steps vs. Accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="steps_count", y="is_correct", hue="model", data=reasoning_df)
    plt.title("Reasoning Steps vs. Accuracy")
    plt.xlabel("Number of Reasoning Steps")
    plt.ylabel("Accuracy (1=Correct, 0=Incorrect)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "reasoning_steps_vs_accuracy.png"))
    plt.close()
    
    # 2. Reasoning Length vs. Confidence
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=reasoning_df, 
        x="reasoning_length", 
        y="confidence",
        hue="model",
        alpha=0.7,
        s=80
    )
    plt.title("Reasoning Length vs. Confidence")
    plt.xlabel("Reasoning Length (characters)")
    plt.ylabel("Confidence Score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "reasoning_length_vs_confidence.png"))
    plt.close()
    
    # 3. Uncertainty Markers vs. Accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="uncertainty_markers", y="is_correct", hue="model", data=reasoning_df)
    plt.title("Uncertainty Markers vs. Accuracy")
    plt.xlabel("Number of Uncertainty Markers")
    plt.ylabel("Accuracy (1=Correct, 0=Incorrect)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "uncertainty_markers_vs_accuracy.png"))
    plt.close()
    
    # 4. Reasoning Time Efficiency (Response Time / Steps)
    reasoning_df["time_per_step"] = reasoning_df["response_time"] / reasoning_df["steps_count"].clip(lower=1)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="time_per_step", hue="is_correct", data=reasoning_df)
    plt.title("Time Efficiency by Model and Correctness")
    plt.xlabel("Model")
    plt.ylabel("Time per Reasoning Step (seconds)")
    plt.yscale("log")  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "time_efficiency_by_model.png"))
    plt.close()

def analyze_confidence_calibration(df):
    """
    Analyze how well calibrated model confidence is compared to actual performance
    using reliability diagrams and calibration metrics.
    """
    print("Analyzing confidence calibration...")
    
    # Create calibration directory
    calibration_dir = os.path.join(OUTPUT_DIR, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Ensure confidence values are available
    df_cal = df.dropna(subset=["confidence", "is_correct"]).copy()
    
    if len(df_cal) == 0:
        print("  ⚠️ No confidence data available for analysis")
        return None
    
    # Normalize confidence to 0-1 range
    df_cal["confidence_norm"] = df_cal["confidence"] / 100
    
    # Create reliability diagrams for each model
    models = df_cal["model"].unique()
    
    calibration_results = []
    
    for model in models:
        model_df = df_cal[df_cal["model"] == model]
        
        # Create reliability diagram
        plt.figure(figsize=(10, 8))
        
        # Create confidence bins (0-0.1, 0.1-0.2, etc.)
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Bin the confidence scores
        model_df["conf_bin"] = pd.cut(
            model_df["confidence_norm"], 
            bins=bin_edges, 
            labels=bin_centers, 
            include_lowest=True
        )
        
        # Calculate accuracy for each bin
        bin_accuracy = model_df.groupby("conf_bin")["is_correct"].mean()
        bin_counts = model_df.groupby("conf_bin").size()
        
        # Plot the reliability diagram
        plt.bar(
            bin_centers, 
            bin_accuracy, 
            width=0.08, 
            alpha=0.5, 
            edgecolor="black",
            color="skyblue"
        )
        
        # Add sample counts above each bar
        for i, (conf, acc) in enumerate(bin_accuracy.items()):
            if pd.notna(conf) and pd.notna(acc):
                count = bin_counts.get(conf, 0)
                plt.text(
                    float(conf), 
                    acc + 0.05, 
                    f"n={count}", 
                    ha="center", 
                    fontsize=8
                )
        
        # Plot the perfect calibration line
        plt.plot([0, 1], [0, 1], "--", color="gray")
        
        plt.title(f"Reliability Diagram for {model}")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.xlim([0, 1])
        plt.ylim([0, 1.1])  # Give room for count labels
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(calibration_dir, f"reliability_{model}.png"))
        plt.close()
        
        # Calculate calibration metrics
        # Expected Calibration Error (ECE)
        binned_stats = model_df.groupby("conf_bin").agg({
            "is_correct": ["mean", "count"],
            "confidence_norm": "mean"
        })
        
        binned_stats.columns = ["accuracy", "count", "confidence"]
        binned_stats = binned_stats.reset_index()
        
        # Calculate ECE (weighted average of |confidence - accuracy|)
        total_samples = binned_stats["count"].sum()
        ece = sum(
            binned_stats["count"] * abs(binned_stats["confidence"] - binned_stats["accuracy"])
        ) / total_samples
        
        # Calculate Brier score (MSE between confidence and outcome)
        brier_score = np.mean((model_df["confidence_norm"] - model_df["is_correct"]) ** 2)
        
        # Save calibration metrics
        calibration_results.append({
            "model": model,
            "ece": ece,
            "brier_score": brier_score,
            "avg_confidence": model_df["confidence_norm"].mean(),
            "avg_accuracy": model_df["is_correct"].mean(),
            "overconfidence": model_df["confidence_norm"].mean() - model_df["is_correct"].mean()
        })
    
    # Save calibration metrics
    cal_df = pd.DataFrame(calibration_results)
    calibration_file = os.path.join(calibration_dir, "calibration_metrics.csv")
    cal_df.to_csv(calibration_file, index=False)
    print(f"✅ Calibration metrics saved to {calibration_file}")
    
    # Create calibration comparison plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="model", y="ece", data=cal_df, color="lightblue")
    sns.scatterplot(x="model", y="overconfidence", data=cal_df, color="red", s=100, label="Overconfidence")
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Expected Calibration Error (ECE) by Model\nLower is Better")
    plt.ylabel("Error / Overconfidence")
    plt.tight_layout()
    plt.savefig(os.path.join(calibration_dir, "calibration_comparison.png"))
    plt.close()
    
    return cal_df

def calculate_additional_metrics():
    """
    Calculate additional metrics:
    - METEOR Score
    - Semantic Similarity
    - Grammar Score
    - Distractor Diversity Score
    """
    print("Calculating additional metrics...")
    
    # Load the original dataset and distractor sets
    original_dataset_file = "mmlu_expanded_dataset.json"
    distractor_files = os.listdir("distractor_sets")
    
    if not os.path.exists(original_dataset_file):
        print(f"⚠️ Original dataset file {original_dataset_file} not found")
        return None
        
    try:
        with open(original_dataset_file, "r", encoding="utf-8") as f:
            original_dataset = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading original dataset: {e}")
        return None
    
    all_metrics = []
    
    # Process each distractor set
    for distractor_file in distractor_files:
        if not distractor_file.endswith(".json"):
            continue
            
        distractor_type = distractor_file.replace("mmlu_distractors_", "").replace(".json", "")
        distractor_file_path = os.path.join("distractor_sets", distractor_file)
        
        try:
            with open(distractor_file_path, "r", encoding="utf-8") as f:
                distractor_set = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading distractor set {distractor_file}: {e}")
            continue
        
        # Match questions between original and distractor set
        for distractor_item in distractor_set:
            question = distractor_item["question"]
            
            # Find matching question in original dataset
            original_item = next((item for item in original_dataset if item["question"] == question), None)
            if not original_item:
                continue
                
            correct_answer = distractor_item["correct_answer"]
            distractors = list(distractor_item["distractors"].values())
            original_options = [opt for opt in original_item["options"] if opt != correct_answer]
            
            # 1. Calculate METEOR Score
            meteor_scores = []
            for orig_opt, new_dist in zip(original_options, distractors):
                # Convert to list of words for METEOR
                reference = [orig_opt.split()]
                hypothesis = new_dist.split()
                try:
                    meteor = meteor_score(reference, hypothesis)
                    meteor_scores.append(meteor)
                except Exception:
                    meteor_scores.append(0)
            
            avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
            
            # 2. Calculate Semantic Similarity
            semantic_similarities = []
            if embedding_model:
                try:
                    # Get embeddings for original and new distractors
                    original_embeddings = embedding_model.encode(original_options)
                    distractor_embeddings = embedding_model.encode(distractors)
                    
                    # Calculate cosine similarity between paired distractors
                    for i in range(min(len(original_embeddings), len(distractor_embeddings))):
                        similarity = np.dot(original_embeddings[i], distractor_embeddings[i]) / (
                            np.linalg.norm(original_embeddings[i]) * np.linalg.norm(distractor_embeddings[i])
                        )
                        semantic_similarities.append(similarity)
                except Exception as e:
                    pass
            
            avg_semantic_similarity = np.mean(semantic_similarities) if semantic_similarities else 0
            
            # 3. Calculate Grammar Score
            grammar_scores = []
            if language_tool:
                for distractor in distractors:
                    try:
                        errors = language_tool.check(distractor)
                        # Calculate grammar score (inversely proportional to number of errors)
                        grammar_score = 1.0 / (1.0 + len(errors))
                        grammar_scores.append(grammar_score)
                    except:
                        grammar_scores.append(0)
            
            avg_grammar_score = np.mean(grammar_scores) if grammar_scores else 0
            
            # 4. Calculate Distractor Diversity Score
            diversity_score = 0
            if embedding_model and len(distractors) > 1:
                try:
                    # Get embeddings for all distractors
                    all_embeddings = embedding_model.encode(distractors)
                    
                    # Calculate pairwise distances
                    pairwise_distances = pdist(all_embeddings, metric='cosine')
                    
                    # Average distance is the diversity score
                    diversity_score = np.mean(pairwise_distances) if len(pairwise_distances) > 0 else 0
                except Exception as e:
                    pass
            
            # Store metrics
            all_metrics.append({
                "question": question,
                "distractor_type": distractor_type,
                "meteor_score": avg_meteor,
                "semantic_similarity": avg_semantic_similarity,
                "grammar_score": avg_grammar_score,
                "diversity_score": diversity_score,
                "subject": distractor_item.get("subject", "unknown")
            })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    metrics_file = os.path.join(METRICS_DIR, "distractor_quality_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✅ Additional metrics saved to {metrics_file}")
    
    # Create aggregated metrics by distractor type
    agg_metrics = metrics_df.groupby("distractor_type").agg({
        "meteor_score": ["mean", "std"],
        "semantic_similarity": ["mean", "std"],
        "grammar_score": ["mean", "std"],
        "diversity_score": ["mean", "std"]
    }).reset_index()
    
    # Flatten column names
    agg_metrics.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in agg_metrics.columns]
    
    # Save aggregated metrics
    agg_file = os.path.join(METRICS_DIR, "aggregated_metrics.csv")
    agg_metrics.to_csv(agg_file, index=False)
    print(f"✅ Aggregated metrics by distractor type saved to {agg_file}")
    
    # Create visualizations
    create_metrics_visualizations(metrics_df, agg_metrics)
    
    return metrics_df, agg_metrics

def create_metrics_visualizations(metrics_df, agg_metrics):
    """Create visualizations for the additional metrics."""
    
    # 1. Bar chart of average metrics by distractor type
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["meteor_score_mean", "semantic_similarity_mean", 
                       "grammar_score_mean", "diversity_score_mean"]
    
    # Extract data for plotting
    plot_data = agg_metrics[["distractor_type"] + metrics_to_plot].melt(
        id_vars=["distractor_type"],
        var_name="metric",
        value_name="value"
    )
    
    # Clean up metric names for display
    plot_data["metric"] = plot_data["metric"].str.replace("_mean", "").str.replace("_", " ").str.title()
    
    # Create plot
    sns.barplot(data=plot_data, x="distractor_type", y="value", hue="metric")
    plt.title("Quality Metrics by Distractor Type")
    plt.xlabel("Distractor Type")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "quality_metrics_by_type.png"))
    plt.close()
    
    # 2. Heatmap of metrics correlation
    plt.figure(figsize=(8, 6))
    # Calculate correlation between metrics
    corr_metrics = metrics_df[["meteor_score", "semantic_similarity", 
                              "grammar_score", "diversity_score"]].corr()
    
    sns.heatmap(corr_metrics, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Quality Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "metrics_correlation.png"))
    plt.close()
    
    # 3. Box plots of diversity scores by distractor type
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_df, x="distractor_type", y="diversity_score")
    plt.title("Distractor Diversity by Type")
    plt.xlabel("Distractor Type")
    plt.ylabel("Diversity Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "diversity_by_type.png"))
    plt.close()

def main():
    """Run the advanced analysis pipeline."""
    print("\n" + "="*50)
    print("🔍 ADVANCED DISTRACTOR ANALYSIS PIPELINE")
    print("="*50 + "\n")
    
    # 1. Load results
    results_data, df = load_results()
    if results_data is None or df is None:
        print("❌ Could not load results. Run enhanced_test.py first.")
        return
        
    print(f"✅ Loaded {len(df)} evaluation results")
    
    # 2. Perform psychometric analysis
    item_df = perform_psychometric_analysis(df)
    
    # 3. Analyze by distractor type
    distractor_stats = analyze_by_distractor_type(df)
    
    # 4. Analyze reasoning patterns
    reasoning_metrics = analyze_reasoning_patterns(df)
    
    # 5. Analyze confidence calibration
    calibration_df = analyze_confidence_calibration(df)
    
    # 6. Calculate additional metrics
    metrics_df, agg_metrics = calculate_additional_metrics()
    
    print("\n" + "="*50)
    print("📊 Analysis complete! Results saved to the evaluation_results directory.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 