import json

# Read the original speeches.json
with open('speeches.json', 'r', encoding='utf-8') as f:
    speeches = json.load(f)

# Select the first 5 elements
sample = speeches[:5]

# Write to speeches-sample.json
with open('speeches-sample.json', 'w', encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)