import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- START: CONFIGURATION & DATA LOADING ---
INPUT_CSV_PATH = 'analyzed_speeches.csv'
POSITIVE_WORDCLOUD_PATH = 'positive_sentiment_wordcloud.png'
NEGATIVE_WORDCLOUD_PATH = 'negative_sentiment_wordcloud.png'

# --- END: CONFIGURATION ---

# 1. Load the data
try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file {INPUT_CSV_PATH} was not found. Please ensure it exists in the current directory.")
    exit()

# 2. Categorize speeches based on sentiment score
print("Categorizing speeches by sentiment...")
positive_speeches = df[df['sentiment_score'] > 0.1]
negative_speeches = df[df['sentiment_score'] < -0.1]

# 3. Clean and count words
def get_word_counts(text_series):
    """Cleans text and returns a Counter of word frequencies."""
    all_words = ' '.join(text_series.dropna()).lower()
    cleaned_words = [
        word for word in all_words.split() 
        if word.isalpha() and word not in stopwords.words('english')
    ]
    return Counter(cleaned_words)

print("Counting words in positive and negative speeches...")
positive_word_counts = get_word_counts(positive_speeches['processed_text'])
negative_word_counts = get_word_counts(negative_speeches['processed_text'])

# Print the top 20 words for each category
print("\nTop 20 most frequent words in POSITIVE speeches:")
for word, count in positive_word_counts.most_common(20):
    print(f"- {word}: {count}")

print("\nTop 20 most frequent words in NEGATIVE speeches:")
for word, count in negative_word_counts.most_common(20):
    print(f"- {word}: {count}")

# 4. Generate word clouds
def generate_wordcloud(word_counts, output_path, title):
    """Generates and saves a word cloud from word counts."""
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        min_font_size=10
    ).generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    print(f"\nWord cloud for '{title}' saved to '{output_path}'.")

# Generate and save the word clouds
generate_wordcloud(positive_word_counts, POSITIVE_WORDCLOUD_PATH, 'Most Frequent Words in Positive Speeches')
generate_wordcloud(negative_word_counts, NEGATIVE_WORDCLOUD_PATH, 'Most Frequent Words in Negative Speeches')

print("\nAdvanced keyword analysis complete!")