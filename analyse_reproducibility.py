#!/usr/bin/env python3
import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_base_filename(json_path):
    """Extract base filename without the numerical suffix"""
    # Match pattern until the timestamp, excluding the -N suffix
    pattern = r'(.*_\d{8}_\d{6}_\d{3})'
    match = re.search(pattern, json_path)
    if match:
        return match.group(1)
    return None

def analyze_json_variability(directory):
    # Find all JSON files
    json_files = glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)
    
    # Group files by their base name
    groups = {}
    for file_path in json_files:
        base_name = extract_base_filename(file_path)
        if base_name:
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(file_path)
    
    # Collect data for each group
    results = []
    for base_name, files in groups.items():
        scores = []
        confidences = []
        phases = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if 'courtship_score' in data and 'confidence' in data:
                    scores.append(data['courtship_score'])
                    confidences.append(data['confidence'])
                    if 'courtship_phase' in data:
                        phases.append(data['courtship_phase'])
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_path}")
                continue
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
        
        # Calculate statistics if we have data
        if scores and confidences:
            short_name = os.path.basename(base_name)
            results.append({
                'base_name': short_name,
                'mean_score': sum(scores) / len(scores),
                'std_score': pd.Series(scores).std(),
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_confidence': sum(confidences) / len(confidences),
                'std_confidence': pd.Series(confidences).std(),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
                'sample_size': len(scores),
                'phases': phases,
                'scores': scores,
                'confidences': confidences
            })
    
    return pd.DataFrame(results)

def plot_variability(df):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Score variability
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='base_name', y='scores', orient='v')
    plt.xticks(rotation=45, ha='right')
    plt.title('Courtship Score Variability')
    plt.ylabel('Score')
    plt.xlabel('')
    plt.tight_layout()
    
    # Plot 2: Confidence variability
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='base_name', y='confidences', orient='v')
    plt.xticks(rotation=45, ha='right')
    plt.title('Confidence Score Variability')
    plt.ylabel('Confidence')
    plt.xlabel('')
    plt.tight_layout()
    
    # Plot 3: Scatter of means with error bars
    plt.subplot(2, 1, 2)
    plt.errorbar(
        df['base_name'], 
        df['mean_score'], 
        yerr=df['std_score'], 
        fmt='o', 
        label='Courtship Score', 
        capsize=5,
        alpha=0.7
    )
    plt.errorbar(
        df['base_name'], 
        df['mean_confidence'], 
        yerr=df['std_confidence'], 
        fmt='s', 
        label='Confidence', 
        capsize=5,
        alpha=0.7
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Mean Scores with Standard Deviation')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('courtship_analysis_variability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table
    summary = df[['base_name', 'mean_score', 'std_score', 'mean_confidence', 
                 'std_confidence', 'sample_size']].sort_values('mean_score', ascending=False)
    
    # Calculate coefficient of variation (CV) for better comparison
    summary['score_cv'] = (summary['std_score'] / summary['mean_score']) * 100
    summary['confidence_cv'] = (summary['std_confidence'] / summary['mean_confidence']) * 100
    
    # Plot CV values
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(summary))
    
    plt.bar([i - bar_width/2 for i in index], summary['score_cv'], bar_width, 
            label='Score CV (%)', alpha=0.7)
    plt.bar([i + bar_width/2 for i in index], summary['confidence_cv'], bar_width, 
            label='Confidence CV (%)', alpha=0.7)
    
    plt.xticks(index, summary['base_name'], rotation=45, ha='right')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Reproducibility: Coefficient of Variation for Scores and Confidence')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('courtship_analysis_cv.png', dpi=300, bbox_inches='tight')
    
    return summary

def main():
    # Directory containing the JSON files
    directory = "output/"  # Update this to your directory
    
    # Analyze the data
    df_exploded = analyze_json_variability(directory)
    
    if df_exploded.empty:
        print("No valid data found for analysis.")
        return
    
    # Plot the results
    summary = plot_variability(df_exploded)
    
    # Print summary statistics
    print("\nSummary Statistics (sorted by mean score):")
    print(summary.to_string(index=False))
    
    # Calculate overall reproducibility metrics
    overall_score_cv = summary['score_cv'].mean()
    overall_conf_cv = summary['confidence_cv'].mean()
    
    print(f"\nOverall Reproducibility Metrics:")
    print(f"Mean Score Coefficient of Variation: {overall_score_cv:.2f}%")
    print(f"Mean Confidence Coefficient of Variation: {overall_conf_cv:.2f}%")
    
    # Save summary to CSV
    summary.to_csv('courtship_analysis_summary.csv', index=False)
    print("\nResults saved to:")
    print("- courtship_analysis_variability.png")
    print("- courtship_analysis_cv.png")
    print("- courtship_analysis_summary.csv")

if __name__ == "__main__":
    main()