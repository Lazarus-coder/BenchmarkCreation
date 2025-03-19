#!/usr/bin/env python3
"""
MMLU Distractor Generation and Evaluation - Enhanced Benchmark Pipeline

This script runs the complete pipeline for the enhanced benchmark:
1. Download expanded MMLU dataset (150 questions from diverse subjects)
2. Generate controlled distractors with taxonomy
3. Evaluate multiple LLM models with confidence scoring and reasoning analysis
4. Perform advanced statistical analysis and psychometric evaluation
"""

import os
import sys
import time
import argparse
import json
from subprocess import run

def check_env():
    """Check if environment variables are set correctly."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is not set.")
        print("   Set it using: export OPENAI_API_KEY=your_key_here")
        return False
    return True

def run_step(cmd, description):
    """Run a step in the pipeline with status reporting."""
    print("\n" + "="*80)
    print(f"🔶 {description}")
    print("="*80)
    
    result = run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Step failed with error code {result.returncode}")
        return False
    
    print(f"✅ {description} completed successfully!")
    return True

def check_dataset():
    """Check if the MMLU dataset was properly downloaded/generated."""
    dataset_file = "mmlu_expanded_dataset.json"
    if not os.path.exists(dataset_file):
        print("⚠️ Dataset file not found. Will generate backup data.")
        return False
        
    try:
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            if len(data) < 5:  # Very few questions
                print("⚠️ Dataset contains too few questions. Will generate backup data.")
                return False
            return True
    except (json.JSONDecodeError, Exception):
        print("⚠️ Dataset file is invalid. Will generate backup data.")
        return False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the enhanced MMLU benchmark pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloading")
    parser.add_argument("--skip-generation", action="store_true", help="Skip distractor generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip advanced analysis")
    parser.add_argument("--use-sample-data", action="store_true", help="Use sample data instead of downloading")
    return parser.parse_args()

def main():
    """Run the complete benchmark pipeline."""
    args = parse_args()
    
    # Check environment
    if not check_env():
        return 1
    
    start_time = time.time()
    
    # Step 1: Download or generate dataset
    if args.use_sample_data:
        print("\n" + "="*80)
        print("🔶 Using sample data instead of downloading")
        print("="*80)
        if not run_step("python generate_test_data.py", "Generating sample MMLU dataset"):
            return 1
    elif not args.skip_download:
        download_success = run_step("python download_mmlu.py", "Downloading expanded MMLU dataset")
        if not download_success or not check_dataset():
            print("\n⚠️ Download failed or dataset invalid. Generating backup data...")
            if not run_step("python generate_test_data.py", "Generating backup MMLU dataset"):
                return 1
    else:
        print("\n⏩ Skipping dataset download")
        if not check_dataset():
            print("⚠️ Valid dataset not found. Generating backup data...")
            if not run_step("python generate_test_data.py", "Generating backup MMLU dataset"):
                return 1
    
    # Step 2: Generate controlled distractors
    if not args.skip_generation:
        if not run_step("python MMLU-DG.py", "Generating controlled distractors"):
            return 1
    else:
        print("\n⏩ Skipping distractor generation")
    
    # Step 3: Evaluate multiple LLMs
    if not args.skip_evaluation:
        if not run_step("python enhanced_test.py", "Evaluating LLM performance"):
            return 1
    else:
        print("\n⏩ Skipping model evaluation")
    
    # Step 4: Perform advanced analysis
    if not args.skip_analysis:
        if not run_step("python advanced_analysis.py", "Running advanced analysis"):
            return 1
    else:
        print("\n⏩ Skipping advanced analysis")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final report
    print("\n" + "="*80)
    print(f"✨ COMPLETE PIPELINE FINISHED SUCCESSFULLY in {total_time:.2f} seconds")
    print("="*80)
    print("\nResults are available in the following directories:")
    print("  - evaluation_results/: Main results directory")
    print("  - evaluation_results/figures/: Visualization plots")
    print("  - evaluation_results/calibration/: Confidence calibration analysis")
    print("  - evaluation_results/anova_results/: Statistical significance tests")
    print("\nKey files to examine:")
    print("  - evaluation_results/enhanced_evaluation_results.json: Complete evaluation data")
    print("  - evaluation_results/psychometric_analysis.csv: Item Response Theory metrics")
    print("  - evaluation_results/distractor_type_analysis.csv: Distractor type effectiveness")
    print("\nTo generate a report: Review the visualizations in the figures directory\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 