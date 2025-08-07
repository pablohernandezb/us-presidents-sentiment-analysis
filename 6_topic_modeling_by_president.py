import pandas as pd
import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim import corpora, models
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['tokens'] = df['processed_text'].apply(preprocess_text)
    return df

def train_lda_model(df, num_topics=12):
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
    return lda_model, corpus, dictionary

def compute_coherence(lda_model, tokens, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def analyze_topics_per_president(df, lda_model, corpus, num_topics):
    df['topic_distribution'] = [lda_model[doc] for doc in corpus]
    topic_matrix = []

    for dist in df['topic_distribution']:
        row = [0.0] * num_topics
        for topic_id, weight in dist:
            row[topic_id] = weight
        topic_matrix.append(row)

    topic_df = pd.DataFrame(topic_matrix, columns=[f'Topic {i}' for i in range(num_topics)])
    df_topics = pd.concat([df[['president']], topic_df], axis=1)

    grouped = df_topics.groupby('president').mean()
    return grouped

def plot_heatmap(topic_dist_by_president):
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(topic_dist_by_president, annot=True, cmap='viridis')

    # Force show all y-axis labels
    ax.set_yticks([i + 0.5 for i in range(len(topic_dist_by_president.index))])
    ax.set_yticklabels(topic_dist_by_president.index, rotation=0, fontsize=9)

    plt.title('Average Topic Distribution per President')
    plt.tight_layout()
    plt.show()

def save_topic_table(topic_dist_by_president, output_path='topic_distribution_by_president.csv'):
    topic_dist_by_president.to_csv(output_path)
    print(f"Saved topic distribution table to: {output_path}")

def main():
    csv_path = 'analyzed_speeches.csv'
    df = load_and_prepare_data(csv_path)

    num_topics = 12
    lda_model, corpus, dictionary = train_lda_model(df, num_topics=num_topics)

    coherence = compute_coherence(lda_model, df['tokens'], dictionary)
    print(f'Coherence Score: {coherence:.4f}')

    # Save visualization
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization_12_topics.html')
    print("LDA visualization saved to lda_visualization_12_topics.html")

    topic_dist_by_president = analyze_topics_per_president(df, lda_model, corpus, num_topics=num_topics)

    plot_heatmap(topic_dist_by_president)
    save_topic_table(topic_dist_by_president)

if __name__ == '__main__':
    main()
