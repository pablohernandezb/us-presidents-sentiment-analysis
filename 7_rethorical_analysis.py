import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from textstat import textstat
import re
import string

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- START: CONFIGURATION & DATA LOADING ---
INPUT_CSV_PATH = 'analyzed_speeches.csv'
OUTPUT_CSV_PATH = 'rhetorical_analysis_results.csv'

# --- END: CONFIGURATION ---

# Load the data
try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file {INPUT_CSV_PATH} was not found. Please ensure it exists in the current directory.")
    exit()

def get_rhetorical_metrics(text_series):
    """Calculates lexical diversity, readability, and n-grams for a series of texts."""
    all_text = ' '.join(text_series.dropna().astype(str))
    
    # 1. Advanced text cleaning and tokenization
    # Remove punctuation
    text_without_punct = all_text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase and split into words
    words = text_without_punct.lower().split()

    # Clean text for n-gram and diversity analysis
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]

    # 2. Lexical Diversity (Type-Token Ratio)
    lexical_diversity = len(set(cleaned_words)) / len(cleaned_words) if cleaned_words else 0
    
    # 3. Readability Score (Flesch-Kincaid Grade Level)
    readability_score = textstat.flesch_kincaid_grade(text_without_punct)
    
    # 4. N-grams (bi-grams and tri-grams)
    bigrams = Counter(ngrams(cleaned_words, 2)).most_common(5)
    trigrams = Counter(ngrams(cleaned_words, 3)).most_common(5)
    
    return pd.Series({
        'Lexical_Diversity': lexical_diversity,
        'Readability_Score': readability_score,
        'Top_5_Bigrams': bigrams,
        'Top_5_Trigrams': trigrams
    })

print("Performing rhetorical analysis on each president's speeches...")

# Group data by president and apply the analysis function
rhetorical_analysis_df = df.groupby('president').apply(lambda x: get_rhetorical_metrics(x['processed_text'])).reset_index()

# Clean up the n-gram columns for saving to CSV
rhetorical_analysis_df['Top_5_Bigrams'] = rhetorical_analysis_df['Top_5_Bigrams'].apply(lambda x: ', '.join([f"'{' '.join(gram)}'" for gram, _ in x]))
rhetorical_analysis_df['Top_5_Trigrams'] = rhetorical_analysis_df['Top_5_Trigrams'].apply(lambda x: ', '.join([f"'{' '.join(gram)}'" for gram, _ in x]))

# Save the results to a CSV file
rhetorical_analysis_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nRhetorical analysis results saved to '{OUTPUT_CSV_PATH}'.")

print("\nRhetorical analysis complete!")