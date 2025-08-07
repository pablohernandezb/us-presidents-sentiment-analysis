import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- START: CONFIGURATION & DATA MAPPING ---
INPUT_CSV_PATH = 'analyzed_speeches.csv'
PLOT_FILE_PATH = 'average_sentiment_by_president.png'

# This dictionary maps all presidents to their political parties
PRESIDENT_TO_PARTY = {
    'George Washington': 'Unaffiliated',
    'John Adams': 'Federalist',
    'Thomas Jefferson': 'Democratic-Republican',
    'James Madison': 'Democratic-Republican',
    'James Monroe': 'Democratic-Republican',
    'John Quincy Adams': 'Democratic-Republican',
    'Andrew Jackson': 'Democrat',
    'Martin Van Buren': 'Democrat',
    'William Harrison': 'Whig',
    'John Tyler': 'Whig',
    'James K. Polk': 'Democrat',
    'Zachary Taylor': 'Whig',
    'Millard Fillmore': 'Whig',
    'Franklin Pierce': 'Democrat',
    'James Buchanan': 'Democrat',
    'Abraham Lincoln': 'Republican',
    'Andrew Johnson': 'Democrat',
    'Ulysses S. Grant': 'Republican',
    'Rutherford B. Hayes': 'Republican',
    'James A. Garfield': 'Republican',
    'Chester A. Arthur': 'Republican',
    'Grover Cleveland': 'Democrat',
    'Benjamin Harrison': 'Republican',
    'William McKinley': 'Republican',
    'Theodore Roosevelt': 'Republican',
    'William Taft': 'Republican',
    'Woodrow Wilson': 'Democrat',
    'Warren G. Harding': 'Republican',
    'Calvin Coolidge': 'Republican',
    'Herbert Hoover': 'Republican',
    'Franklin D. Roosevelt': 'Democrat',
    'Harry S. Truman': 'Democrat',
    'Dwight D. Eisenhower': 'Republican',
    'John F. Kennedy': 'Democrat',
    'Lyndon B. Johnson': 'Democrat',
    'Richard M. Nixon': 'Republican',
    'Gerald Ford': 'Republican',
    'Jimmy Carter': 'Democrat',
    'Ronald Reagan': 'Republican',
    'George H. W. Bush': 'Republican',
    'Bill Clinton': 'Democrat',
    'George W. Bush': 'Republican',
    'Barack Obama': 'Democrat',
    'Donald Trump': 'Republican',
    'Joe Biden': 'Democrat'
}

# This dictionary maps presidents to their years of presidency
PRESIDENCY_YEARS = {
    'George Washington': '(1789-1797)',
    'John Adams': '(1797-1801)',
    'Thomas Jefferson': '(1801-1809)',
    'James Madison': '(1809-1817)',
    'James Monroe': '(1817-1825)',
    'John Quincy Adams': '(1825-1829)',
    'Andrew Jackson': '(1829-1837)',
    'Martin Van Buren': '(1837-1841)',
    'William Harrison': '(1841-1841)',
    'John Tyler': '(1841-1845)',
    'James K. Polk': '(1845-1849)',
    'Zachary Taylor': '(1849-1850)',
    'Millard Fillmore': '(1850-1853)',
    'Franklin Pierce': '(1853-1857)',
    'James Buchanan': '(1857-1861)',
    'Abraham Lincoln': '(1861-1865)',
    'Andrew Johnson': '(1865-1869)',
    'Ulysses S. Grant': '(1869-1877)',
    'Rutherford B. Hayes': '(1877-1881)',
    'James A. Garfield': '(1881-1881)',
    'Chester A. Arthur': '(1881-1885)',
    'Grover Cleveland': '(1885-1889)',
    'Benjamin Harrison': '(1889-1893)',
    'Grover Cleveland': '(1893-1897)',
    'William McKinley': '(1897-1901)',
    'Theodore Roosevelt': '(1901-1909)',
    'William Taft': '(1909-1913)',
    'Woodrow Wilson': '(1913-1921)',
    'Warren G. Harding': '(1921-1923)',
    'Calvin Coolidge': '(1923-1929)',
    'Herbert Hoover': '(1929-1933)',
    'Franklin D. Roosevelt': '(1933-1945)',
    'Harry S. Truman': '(1945-1953)',
    'Dwight D. Eisenhower': '(1953-1961)',
    'John F. Kennedy': '(1961-1963)',
    'Lyndon B. Johnson': '(1963-1969)',
    'Richard M. Nixon': '(1969-1974)',
    'Gerald Ford': '(1974-1977)',
    'Jimmy Carter': '(1977-1981)',
    'Ronald Reagan': '(1981-1989)',
    'George H. W. Bush': '(1989-1993)',
    'Bill Clinton': '(1993-2001)',
    'George W. Bush': '(2001-2009)',
    'Barack Obama': '(2009-2017)',
    'Donald Trump': '(2017-2021)',
    'Joe Biden': '(2021-Present)'
}

# Define a color palette for the parties for better visualization
PARTY_COLORS = {
    'Republican': 'red',
    'Democrat': 'blue',
    'Federalist': 'gray',
    'Democratic-Republican': 'darkgreen',
    'Whig': 'purple',
    'Unaffiliated': 'black'
}
# --- END: CONFIGURATION ---

# Load the analyzed data from the provided CSV file
try:
    df = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file {INPUT_CSV_PATH} was not found. Please ensure the file exists in the current directory.")
    exit()

print("Generating the average sentiment bar chart...")

# Add political party information to the DataFrame
df['political_party'] = df['president'].map(PRESIDENT_TO_PARTY)

# Calculate the average sentiment score per president
average_sentiment = df.groupby(['president', 'political_party'])['sentiment_score'].mean().reset_index()

# Add the presidency years to the DataFrame for formatting
average_sentiment['presidency_years'] = average_sentiment['president'].map(PRESIDENCY_YEARS)

# Combine president name and years for the x-axis labels
average_sentiment['president_label'] = average_sentiment['president'] + ' ' + average_sentiment['presidency_years']

# Sort the values for better visualization
average_sentiment.sort_values(by='sentiment_score', ascending=False, inplace=True)

# Set the style and create the plot
sns.set_style("whitegrid")
plt.figure(figsize=(15, 8))

# Create the bar plot with a custom color palette
ax = sns.barplot(
    x='president_label',
    y='sentiment_score',
    hue='political_party',
    data=average_sentiment,
    palette=PARTY_COLORS,
    dodge=False,
    order=average_sentiment['president_label']
)

# Add titles and labels for clarity
plt.title('Average Sentiment Score by President', fontsize=16)
plt.xlabel('President (Years of Presidency)', fontsize=12)
plt.ylabel('Average VADER Compound Sentiment Score', fontsize=12)

# Move the legend outside the plot area
plt.legend(title='Political Party', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add the sentiment score numbers on each bar
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        xytext=(0, -10),
        textcoords='offset points',
        color='white',
        rotation=90,
        fontweight='bold'
    )

# Rotate x-axis labels for readability
plt.xticks(rotation=90)

# Adjust layout to make space for the legend and remove excess whitespace
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(PLOT_FILE_PATH)
plt.close()

print(f"\nThe average sentiment bar chart has been created and saved as '{PLOT_FILE_PATH}'.")