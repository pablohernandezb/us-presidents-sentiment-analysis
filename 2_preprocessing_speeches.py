import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import pandas as pd
import sys

# === START: CONFIGURATION - UPDATE THESE VARIABLES ===
# Replace these strings with the exact key names from your speeches.json file
JSON_FILE_PATH = 'speeches.json'
PRESIDENT_KEY = 'president'
DATE_KEY = 'date'
TEXT_KEY = 'transcript'
DOC_NAME_KEY = 'doc_name'
TITLE_KEY = 'title'
# === END: CONFIGURATION ===

# Make sure you've downloaded the necessary NLTK data and spaCy model
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    sys.exit()

def preprocess_text(text):
    """
    Cleans and prepares text data for sentiment analysis.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized_tokens)

# Main script execution
output_csv_path = 'preprocessed_speeches.csv'

try:
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        speeches_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file {JSON_FILE_PATH} was not found.")
    sys.exit()

preprocessed_results = []

for speech in speeches_data:
    try:
        president_name = speech.get(PRESIDENT_KEY, 'Unknown')
        speech_date = speech.get(DATE_KEY, 'Unknown')
        full_text = speech.get(TEXT_KEY, '')
        doc_name = speech.get(DOC_NAME_KEY, 'Unknown')
        speech_title = speech.get(TITLE_KEY, 'Unknown')

        if full_text and isinstance(full_text, str) and full_text.strip():
            processed_text = preprocess_text(full_text)
        else:
            processed_text = ""

        preprocessed_results.append({
            'president': president_name,
            'date': speech_date,
            'doc_name': doc_name,
            'title': speech_title,
            'processed_text': processed_text
        })
    except Exception as e:
        print(f"Skipping an entry due to an error: {e}")

df = pd.DataFrame(preprocessed_results)
df.to_csv(output_csv_path, index=False)

print(f"\nPreprocessing complete! The cleaned data has been saved to '{output_csv_path}'.")
print(f"A total of {len(df)} speeches were processed.")