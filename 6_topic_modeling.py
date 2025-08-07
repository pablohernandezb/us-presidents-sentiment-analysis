import pandas as pd
import nltk
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def compute_coherence(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=42,
                                passes=10,
                                alpha='auto')
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def run_topic_modeling():
    # Load dataset
    df = pd.read_csv("analyzed_speeches.csv")
    texts = df['processed_text'].dropna().apply(preprocess).tolist()

    # Build dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Optimize number of topics
    print("Finding optimal number of topics...")
    model_list, coherence_values = compute_coherence(dictionary, corpus, texts, start=2, limit=15, step=1)

    # Plot coherence
    plt.plot(range(2, 15), coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.title("Optimal Number of Topics")
    plt.savefig("coherence_scores.png")
    plt.close()

    # Best model
    best_idx = coherence_values.index(max(coherence_values))
    optimal_model = model_list[best_idx]
    num_topics = 2 + best_idx

    print(f"Best number of topics: {num_topics}")
    print(f"Coherence Score: {coherence_values[best_idx]:.4f}")

    # Save visualization
    vis = gensimvis.prepare(optimal_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("LDA visualization saved to lda_visualization.html")

if __name__ == "__main__":
    run_topic_modeling()
