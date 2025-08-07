import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- START: CONFIGURATION ---
INPUT_CSV_PATH = 'analyzed_speeches.csv'
OUTPUT_DIR = 'individual_sentiment_plots'
# --- END: CONFIGURATION ---

try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file {INPUT_CSV_PATH} was not found. Please ensure the sentiment analysis script ran successfully.")
    exit()

# Create an output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Generating individual plots for each president...")

# Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows with invalid dates and sort by date
df = df.dropna(subset=['date'])
df.sort_values(by='date', inplace=True)

# Get a list of unique presidents
presidents = df['president'].unique()

# Loop through each president and create a plot
for president in presidents:
    # Filter the DataFrame for the current president
    president_df = df[df['president'] == president]
    
    # Set the style and create a figure
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot the sentiment scores over time
    sns.lineplot(data=president_df, x='date', y='sentiment_score', marker='o')
    
    # Add titles and labels
    plt.title(f'Sentiment Trends for {president}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('VADER Compound Sentiment Score', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1) # Add a horizontal line at 0
    
    # Save the plot to a file
    filename = f"{president.replace(' ', '_')}_sentiment.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    
    print(f"Created plot for {president} and saved to '{os.path.join(OUTPUT_DIR, filename)}'")

print("\nAll individual plots have been generated.")