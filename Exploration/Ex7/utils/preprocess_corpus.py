import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import numpy as np 


# 1. Define Contractions Dictionary (Normalization)
contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I could have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i could have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it could have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she could have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that could have", "that's": "that is", "there'd": "there would", "there'd've": "there could have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they could have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we could have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all could have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you could have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def preprocess_sentence(sentence, remove_stopwords=True):
    """
    Clean and normalize a single sentence.
    
    Args:
        sentence (str): The raw text to be cleaned.
        remove_stopwords (bool): Whether to remove NLTK stopwords. 
                                 Set to False for 'headlines' to keep linguistic flow.
    """
    # 1. Lowercase and remove HTML tags using BeautifulSoup
    sentence = sentence.lower()
    sentence = BeautifulSoup(sentence, "lxml").text 
    
    # 2. Remove parentheticals and special characters using Regex
    sentence = re.sub(r'\([^)]*\)', '', sentence) # Remove text in (brackets)
    sentence = re.sub('"','', sentence) # Remove double quotes
    
    # 3. Text Normalization (Expanding Contractions)
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")])
    
    # 4. Remove 's and filter non-alphabetic characters
    sentence = re.sub(r"'s\b","", sentence)
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    
    # 5. Stopword Removal
    if remove_stopwords:
        # For 'text' column: remove stopwords to focus on important keywords
        stop_words = set(stopwords.words('english'))
        tokens = ' '.join([word for word in sentence.split() if not word in stop_words if len(word) > 1])
    else:
        # For 'headlines' column: keep stopwords for natural sentence structure
        tokens = ' '.join([word for word in sentence.split() if len(word) > 1])
    
    return tokens

# Usage Example:
# clean_text = [preprocess_sentence(s) for s in data['text']]
# clean_headlines = [preprocess_sentence(s, remove_stopwords=False) for s in data['headlines']]




# utils/preprocess_corpus.py 
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_corpus(df):
    """
    Apply preprocessing to the entire news dataset.
    - Text: remove stopwords
    - Headlines: keep stopwords
    """
    print("Starting corpus preprocessing...")
    df['text'] = df['text'].apply(lambda x: preprocess_sentence(x, remove_stopwords=True))
    df['headlines'] = df['headlines'].apply(lambda x: preprocess_sentence(x, remove_stopwords=False))
    
    # Drop potential empty strings after preprocessing
    df.replace('', np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    
    print(f"Preprocessing complete. Remaining samples: {len(df)}")
    return df