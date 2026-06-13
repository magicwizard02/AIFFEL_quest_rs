# utils/analyze_corpus.py
import matplotlib.pyplot as plt
import numpy as np

def analyze_corpus(df, column, name="Column", percentiles=[80, 90], ax=None):
    """
    General purpose function to analyze word count distribution.
    - ax: If provided, plots on the specific subplot axis.
    """
    # Calculate word counts
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
    # If ax is None, create a new figure; otherwise, use the passed ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(word_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='white')
    
    # Add Dotted Vertical Lines for Percentiles
    colors = ['orange', 'red', 'green']
    for i, p in enumerate(percentiles):
        val = stats[f'{p}th %']
        ax.axvline(val, color=colors[i % len(colors)], linestyle=':', linewidth=2, 
                   label=f'{p}th Percentile ({int(val)})')

    ax.set_title(f'Distribution: {name}', fontsize=14)
    ax.set_xlabel('Number of Words')
    ax.set_ylabel('Samples')
    ax.legend()
    
    return stats


def get_vocab_size(tokenizer, threshold=7):
    """
    Analyzes word frequencies in the tokenizer and returns the 
    optimal vocabulary size excluding rare words.
    """
    total_cnt = len(tokenizer.word_index) # Number of unique words
    rare_cnt = 0 # Words with frequency less than threshold
    total_freq = 0 # Total frequency of all words
    rare_freq = 0 # Total frequency of rare words

    # Iterate through word counts to calculate rare word statistics
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('--- Tokenizer Analysis ---')
    print('Total vocabulary size:', total_cnt)
    print(f'Number of rare words (freq < {threshold}): {rare_cnt}')
    print(f"Percentage of rare words in vocab: {(rare_cnt / total_cnt)*100:.2f}%")
    print(f"Total frequency share of rare words: {(rare_freq / total_freq)*100:.2f}%")
    
    # +1 is for the padding token (index 0)
    return total_cnt - rare_cnt + 1


def analyze_threshold(tokenizer, threshold=7):
    """
    Analyzes how many words fall below a certain frequency threshold.
    Helps in deciding the optimal vocabulary size for the model.
    """
    # Total number of unique words in the tokenizer's dictionary
    total_cnt = len(tokenizer.word_index) 
    
    # Counter for words with frequency less than the threshold
    rare_cnt = 0 
    
    # Sum of all word appearances in the entire dataset
    total_freq = 0 
    
    # Sum of appearances of only the rare words
    rare_freq = 0 

    # word_counts contains pairs of (word, count)
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # If the word frequency is less than the user-defined threshold
        if value < threshold:
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('--- Vocabulary Impact Analysis ---')
    print(f"Total Unique Words (Vocab): {total_cnt}")
    print(f"Number of rare words (freq < {threshold}): {rare_cnt}")
    
    # Calculate percentage of rare words compared to total vocab
    print(f"Rare words ratio in Vocab: {(rare_cnt / total_cnt)*100:.2f}%")
    
    # Calculate how much of the actual text data consists of these rare words
    # If this % is low, it's safe to remove them.
    print(f"Rare words frequency ratio in Data: {(rare_freq / total_freq)*100:.2f}%")
    print('----------------------------------')
    
    # Recommended vocabulary size excluding rare words (+2 for [PAD] and [UNK])
    return total_cnt - rare_cnt + 2

def find_threshold_by_coverage(tokenizer, target_coverage=0.97):
    """
    Finds the minimum frequency threshold required to maintain a specific 
    percentage of the total word occurrences (data coverage).
    
    Args:
        tokenizer: The fitted Keras Tokenizer instance.
        target_coverage: The fraction of total data to preserve (0.97 = 97%).
        
    Returns:
        threshold: The frequency value to use for filtering rare words.
    """
    # 1. Extract all word frequencies and sort them in descending order (highest to lowest)
    frequencies = sorted(tokenizer.word_counts.values(), reverse=True)
    
    # 2. Calculate the total sum of all word occurrences in the corpus
    total_word_count = sum(frequencies)
    
    # 3. Iterate through sorted frequencies to find the cutoff point
    cumulative_count = 0
    threshold = 1
    
    for freq in frequencies:
        cumulative_count += freq
        # Check if we have reached the desired coverage percentage
        if (cumulative_count / total_word_count) >= target_coverage:
            threshold = freq
            break
            
    print(f">>> Target Coverage: {target_coverage*100}%")
    print(f">>> Calculated Optimal Threshold: {threshold}")
    
    return threshold