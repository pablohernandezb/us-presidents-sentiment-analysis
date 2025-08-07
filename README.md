# US Presidents Sentiment & Rhetorical Analysis

This repository is a sandbox project designed to deepen my skills in machine learning, natural language processing (NLP), and large language models (LLMs) through hands-on research and experimentation. The project analyzes US presidential speeches to uncover sentiment trends, rhetorical patterns, and linguistic features.

## Research & Learning Objectives

- **Research Focus:**  
  - Explore the evolution of presidential rhetoric and sentiment over time.
  - Compare traditional NLP techniques with LLM-based approaches for text analysis.
  - Investigate the relationship between sentiment, lexical diversity, and readability in political speech.

- **Learning Goals:**  
  - Gain practical experience with text preprocessing, sentiment analysis, and rhetorical metrics.
  - Integrate and evaluate LLMs for advanced text understanding.
  - Develop robust data visualization and reporting skills.

## Data Source

All speech data is sourced from the [Miller Center](https://millercenter.org/the-presidency/presidential-speeches) at the University of Virginia, which provides a comprehensive, non-partisan collection of US presidential speeches.

## Features

- **Data Collection & Preprocessing:**  
  - Aggregates and cleans historical US presidential speeches.
  - Handles text normalization, tokenization, and stopword removal.

- **Sentiment Analysis:**  
  - Calculates sentiment scores for each speech using both classical NLP and LLM-based approaches.
  - Visualizes sentiment trends across different presidencies.

- **Keyword & Word Cloud Analysis:**  
  - Extracts and visualizes the most frequent positive and negative keywords.
  - Generates word clouds for quick insights.

- **Rhetorical Analysis:**  
  - Computes lexical diversity, readability scores, and n-gram statistics (bigrams, trigrams).
  - Groups and compares rhetorical metrics by president.

- **LLM Integration:**  
  - Utilizes large language models for deeper contextual sentiment and rhetorical analysis (optional, see `llm_analysis.py`).
  - Compares LLM-based results with traditional methods.

## Project Structure

```
us-presidents-sentiment-analysis/
│
├── analyzed_speeches.csv                #  Processed speeches with sentiment and other metrics
├── speeches.json                        #  Raw or preprocessed speeches data from Miller Center
├── speeches-sample.json                 #  Sample of speeches for quick testing or sharing
├── 1_download_mc_speeches.py            #  Script to download speeches from Miller Center
├── 2_preprocessing_speeches.py          #  Script for cleaning and preprocessing speech text
├── 3_sentiment_analysis.py              #  Script for basic sentiment analysis
├── 4_visualize_avg_sentiment_by_party.py              #  Script for visualizing avg sentiment analysis by party 
├── 4_visualize_avg_sentiment_by_president.py              #  Script for visualizing avg sentiment analysis by president
├── 5_advance_sentiment_analysis.py      #  Advanced sentiment and keyword analysis, word clouds
├── 6_topic_modeling.py                  #  Topic modeling across all speeches
├── 6_topic_modeling_by_president.py     #  Topic modeling grouped by president
├── 7_rethorical_analysis.py             #  Rhetorical metrics: lexical diversity, readability, n-grams
├── requirements.txt                     #  Python dependencies for the project
├── plots/                               #  Folder for generated plots and visualizations (word clouds, charts, etc.)
└── README.md                            #  Project documentation (this file)
```

## Example Outputs

- **Word Clouds:**  
  ![Positive Word Cloud](positive_sentiment_wordcloud.png)  
  ![Negative Word Cloud](negative_sentiment_wordcloud.png)

- **Rhetorical Metrics:**  
  Results saved in `rhetorical_analysis_results.csv` with columns for lexical diversity, readability, and top n-grams.

## Python 3.11 Setup & Dependencies

This project is developed and tested with **Python 3.11**.  
It is recommended to use a virtual environment to manage dependencies.

### 1. Install Python 3.11

- Download and install Python 3.11 from the [official website](https://www.python.org/downloads/release/python-3110/).
- Ensure `python` and `pip` point to Python 3.11 in your terminal:
  ```sh
  python --version
  # Output should be: Python 3.11.x
  ```

### 2. Create and Activate a Virtual Environment

On Windows:
```sh
python -m venv .venv311
.venv311\Scripts\activate
```

On macOS/Linux:
```sh
python3 -m venv .venv311
source .venv311/bin/activate
```

### 3. Install Required Modules

Install all dependencies using:
```sh
pip install -r requirements.txt
```

If you need to generate `requirements.txt`, you can do so with:
```sh
pip freeze > requirements.txt
```

---

*Now you are ready to run the analysis scripts as described below!*

## How to Perform the Analysis

1. **Clone the repository:**
   ```sh
   git clone https://github.com/<your-username>/us-presidents-sentiment-analysis.git
   cd us-presidents-sentiment-analysis
   ```

2. **Download the speeches**
   ```sh
   python 1_download_mc_speeches.py
   ```

3. **Preprocessing the speeches**
   ```sh
   python 2_preprocessing_speeches.py
   ```

4. **Sentiment analysis**
   ```sh
   python 3_sentiment_analysis.py
   ```

5. **Visualize average sentiment analysis**
   ```sh
   4_visualize_avg_sentiment_by_party.py
   4_visualize_avg_sentiment_by_president.py
   ```


6. **Run sentiment and keyword analysis:**
   ```sh
   python 5_advance_sentiment_analysis.py
   ```

7. **Run topic modeling:**
    ```sh
    python 6_topic_modeling.py
    python 6_topic_modeling_by_president.py
    ```

8. **Run rhetorical analysis:**
   ```sh
   python 7_rethorical_analysis.py
   ```

## Methodology and Metrics 

1. Sentiment Analysis

Using the VADER sentiment analysis tool, this project calculates a compound sentiment score for each speech. The analysis reveals how the emotional tone of presidential communication has evolved over time. Key visualizations include:

Average Sentiment by President: A bar chart that ranks each president by the average sentiment score of their speeches, color-coded by political party.

Word Clouds: Visual representations of the most frequently used words in both positive and negative speeches, providing insight into the topics and themes associated with different emotional tones.

2. Rhetorical Analysis

The rhetorical analysis provides a quantitative look at each president's communication style. This is achieved by calculating and comparing several metrics:

Lexical Diversity: Measures the richness and variety of vocabulary used. A higher score indicates a more diverse word choice, while a lower score suggests a more repetitive, perhaps simpler, style.

Readability (Flesch-Kincaid Grade Level): Estimates the education level required to understand the speeches. A lower score indicates a more accessible, conversational style, while a higher score points to a more complex, formal style.

Top N-Grams: Identifies the most common two-word and three-word phrases used by each president, revealing signature rhetorical patterns and key policy focuses.

The results of this analysis are saved in rhetorical_analysis_results.csv, providing a structured overview of each president's communication style.

## Skills Demonstrated

- **Machine Learning & NLP:**  
  - Sentiment analysis using both rule-based and LLM approaches.
  - Text preprocessing, tokenization, and feature extraction.

- **Data Visualization:**  
  - Word clouds and frequency analysis for interpretability.

- **Rhetorical & Linguistic Analysis:**  
  - Lexical diversity, readability, and n-gram statistics.

- **Python & Data Science Tools:**  
  - pandas, nltk, wordcloud, matplotlib, textstat, and integration with LLM APIs.

## License

MIT License

---

*This project is a sandbox for learning and research. For questions or collaboration, please open an issue or contact me!*