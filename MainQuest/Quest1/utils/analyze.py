import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_corpus(df, column, name="Column", percentiles=[80, 90, 95, 99]):
    """
    Analyzes the distribution of token/word counts in a specific dataframe column.
    This is used to determine the optimal 'max_len' for the GPT model.
    
    Args:
        df (pd.DataFrame): The dataset containing text.
        column (str): The column name to analyze (e.g., 'question' or 'answer').
        name (str): Display name for the plot title.
        percentiles (list): Percentile markers to display.
        
    Returns:
        stats (dict): Dictionary containing mean, max, min, and percentile values.
    """
    # Calculate word counts (or token counts if already tokenized)
    # For raw text, we split by whitespace as a proxy for tokens.
    word_counts = df[column].apply(lambda x: len(str(x).split()))
    
    # Calculate Statistics
    stats = {
        'mean': np.mean(word_counts),
        'max': np.max(word_counts),
        'min': np.min(word_counts),
    }
    
    # Calculate Percentiles
    for p in percentiles:
        stats[f'{p}th %'] = np.percentile(word_counts, p)

    # Plotting Logic
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='white')
    
    # Add Vertical Lines for Percentiles
    colors = ['orange', 'red', 'green', 'purple']
    for i, p in enumerate(percentiles):
        val = stats[f'{p}th %']
        plt.axvline(val, color=colors[i % len(colors)], linestyle=':', linewidth=2, 
                   label=f'{p}th Percentile ({int(val)})')

    plt.title(f'Sequence Length Distribution: {name}', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words/Tokens')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return stats

def find_threshold_by_coverage(frequencies, target_coverage=0.97):
    """
    Finds the frequency threshold required to maintain a specific 
    percentage of total word occurrences. Useful if you are building 
    a custom vocabulary or refining a SentencePiece model.
    
    Args:
        frequencies (list/dict_values): A list of word frequencies.
        target_coverage (float): The fraction of total data to preserve (e.g., 0.97).
        
    Returns:
        threshold (int): The frequency value to use for filtering rare words.
    """
    # Sort frequencies in descending order
    sorted_freqs = sorted(frequencies, reverse=True)
    
    total_word_count = sum(sorted_freqs)
    cumulative_count = 0
    threshold = 1
    
    for freq in sorted_freqs:
        cumulative_count += freq
        if (cumulative_count / total_word_count) >= target_coverage:
            threshold = freq
            break
            
    print(f"[Analysis] Target Coverage: {target_coverage*100}%")
    print(f"[Analysis] Calculated Optimal Frequency Threshold: {threshold}")
    
    return threshold