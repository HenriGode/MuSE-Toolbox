import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def parse_run_name(run_name):
    # Determine dataset
    dataset = None
    if 'circ_8ch' in run_name:
        dataset = 'circ_8ch'
    elif 'circ_2-8ch' in run_name:
        dataset = 'circ_2-8ch'
    elif 'rand_8ch' in run_name:
        dataset = 'rand_8ch'
    elif 'rand_2-8ch' in run_name:
        dataset = 'rand_2-8ch'
    else:
        dataset = 'unknown'

    # Determine feature
    feature = None
    if 'purestft' in run_name or 'pure_stft' in run_name:
        feature = 'stft'
    elif 'logmel' in run_name or 'log_mel' in run_name:
        feature = 'logmel'
    elif 'wgmsc' in run_name:
        feature = 'wgmsc'
    elif 'gmsc' in run_name:
        feature = 'gmsc'
    elif 'ipd' in run_name:
        feature = 'ipd'
    else:
        feature = 'unknown'

    return dataset, feature

def main():

    arichtecture = "gru"
    base_dir = Path("/data4/Henri/MuSE-Toolbox/outputs/source_counting/J3_feature_comparison")
    
    # Find all runs for the specified architecture (handling trailing underscores)
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d.name.endswith(arichtecture) or d.name.endswith(f"{arichtecture}_"))]
    
    all_data = []
    
    for run_dir in run_dirs:
        dataset, feature = parse_run_name(run_dir.name)
        if dataset == 'unknown' or feature == 'unknown':
            print(f"Could not parse dataset or feature for {run_dir.name}")
            continue
            
        # Find all timestamp directories
        timestamp_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            print(f"No timestamp directory found in {run_dir}")
            continue
            
        # Sort by name (which is timestamp format YYYY-MM-DD_HH-MM-SS) in descending order
        sorted_timestamp_dirs = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)
        
        metrics_file = None
        for timestamp_dir in sorted_timestamp_dirs:
            potential_file = timestamp_dir / "test" / "results" / "test_metrics.csv"
            if potential_file.exists():
                metrics_file = potential_file
                break
                
        if metrics_file is None:
            print(f"Metrics file not found in any timestamp dir for: {run_dir.name}")
            continue
            
        # Read the metrics file
        df_metrics = pd.read_csv(metrics_file)
        
        if 'Accuracy' not in df_metrics.columns or 'MAE' not in df_metrics.columns:
            print(f"Accuracy or MAE not found in {metrics_file}")
            continue
            
        # Extract the metrics and append dataset and feature info
        for _, row in df_metrics.iterrows():
            all_data.append({
                'Dataset': dataset,
                'Feature': feature,
                'Accuracy': row['Accuracy'],
                'MAE': row['MAE']
            })
            
    if not all_data:
        print("No data was collected!")
        return
        
    df_all = pd.DataFrame(all_data)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Feature order
    feature_order = ['stft', 'logmel', 'ipd', 'gmsc', 'wgmsc']
    dataset_order = ['circ_8ch', 'circ_2-8ch', 'rand_8ch', 'rand_2-8ch']
    
    # Accuracy plot
    sns.violinplot(
        data=df_all, 
        x="Feature", 
        y="Accuracy", 
        hue="Dataset", 
        ax=axes[0],
        order=feature_order,
        hue_order=dataset_order,
        palette="muted",
        cut=0,
        inner="quartile"
    )
    axes[0].set_title("Accuracy by Feature and Dataset", fontsize=16)
    axes[0].set_ylabel("Accuracy", fontsize=14)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Dataset")
    
    # MAE plot
    sns.violinplot(
        data=df_all, 
        x="Feature", 
        y="MAE", 
        hue="Dataset", 
        ax=axes[1],
        order=feature_order,
        hue_order=dataset_order,
        palette="muted",
        cut=0,
        inner="quartile"
    )
    axes[1].set_title("Mean Absolute Error (MAE) by Feature and Dataset", fontsize=16)
    axes[1].set_ylabel("MAE", fontsize=14)
    axes[1].set_xlabel("Feature", fontsize=14)
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Dataset")
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f"results_visualization_{arichtecture}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Optional: Save the combined dataframe to csv
    df_all.to_csv(Path(__file__).parent / f"combined_metrics_{arichtecture}.csv", index=False)
    print("Combined dataframe saved as well.")

if __name__ == "__main__":
    main()
