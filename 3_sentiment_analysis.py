import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using VADER.
    Returns the compound sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    if isinstance(text, str) and text.strip():
        return analyzer.polarity_scores(text)['compound']
    return 0.0

# --- START: CONFIGURATION ---
INPUT_CSV_PATH = 'preprocessed_speeches.csv'
OUTPUT_CSV_PATH = 'analyzed_speeches.csv'
# --- END: CONFIGURATION ---

try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file {INPUT_CSV_PATH} was not found.")
    exit()

# Apply sentiment analysis to the 'processed_text' column
print("Analyzing sentiment for each speech...")
df['sentiment_score'] = df['processed_text'].apply(analyze_sentiment)

# Save the updated DataFrame to a new CSV file
df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\nSentiment analysis complete!")
print(f"The results have been saved to '{OUTPUT_CSV_PATH}'.")
print("You can now begin to visualize your data.")