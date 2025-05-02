# Example: Testing sample texts for all five readability categories using textstat
import textstat
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Sample texts for each category
texts = {
    "Very Easy (Elementary)": """
I love the zoo. Lions are big. They roar. Monkeys climb trees. They like bananas. The elephant is big. It sprays water. Giraffes have long necks. They eat leaves. I eat ice cream. Mom smiles. The zoo is great.
""",
    "Easy (Middle School)": """
Drawing is my favorite activity. I use colorful crayons, sharp pencils, and bright markers to create art. I often draw pictures of playful kittens, loyal dogs, or shady trees. Sometimes, I sketch my cheerful family at home. Drawing helps me relax and feel joyful. I practice every afternoon in my quiet room. My kind teacher always praises my creative pictures. Recently, I drew a happy puppy with a fluffy tail. My classmates loved the drawing. Someday, I want to be a talented artist. Drawing brings me happiness every day.
""",
    "Moderate (High School)": """
The American Revolution was a key event. It started in 1775 when colonists wanted freedom from Britain. Taxes, like the Stamp Act, made them angry. In 1776, they wrote the Declaration of Independence to explain their fight. George Washington led the army with bravery. Battles at Lexington and Yorktown were important. In 1783, the Treaty of Paris ended the war. America became a free nation. The revolution encouraged other countries to seek liberty.
""",
    "Difficult (College)": """
Climate change threatens global ecosystems. Rising temperatures from greenhouse emissions cause polar ice to melt and sea levels to rise. Coastal regions face flooding, while agriculture suffers from erratic weather. Biodiversity declines, especially in tropics. Mitigation, like renewable energy and carbon taxes, is vital. Yet, global cooperation is uneven, slowing progress. Scientists warn of irreversible damage without urgent action. Policy changes and education are crucial for sustainability.
""",
    "Very Difficult (Specialist)": """
Quantum computing revolutionizes computational paradigms. Unlike classical systems, quantum computers exploit superposition, entanglement, and quantum interference to achieve exponential computational speed. Qubits, existing in superposed states, enable massive parallelism. Advances in error correction and coherence time mitigate decoherence, stabilizing systems. Applications include cryptographic analysis, complex optimization, and molecular simulations. Scalability remains challenging, requiring interdisciplinary efforts from physicists, engineers, and computer scientists to achieve practical quantum supremacy.
"""
}

# Define thresholds for readability metrics (five tiers)
thresholds = {
    'flesch_kincaid_grade': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 13},
    'flesch_reading_ease': {'very_easy': 80, 'easy': 60, 'moderate': 40, 'difficult': 20},
    'gunning_fog': {'very_easy': 5, 'easy': 8, 'moderate': 11, 'difficult': 14},
    'smog_index': {'very_easy': 6, 'easy': 9, 'moderate': 11, 'difficult': 13},
    'automated_readability_index': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 14}
}

# Function to categorize a readability score


def categorize_score(metric, value):
    drifted_value = value
    if metric == 'flesch_reading_ease' and value < 0:
        drifted_value = 0
    if metric not in thresholds:
        return "N/A"
    t = thresholds[metric]
    if metric == 'flesch_reading_ease':  # Inverted scale: higher is easier
        if drifted_value > t['very_easy']:
            return "Very Easy (Elementary)"
        elif drifted_value > t['easy']:
            return "Easy (Middle School)"
        elif drifted_value > t['moderate']:
            return "Moderate (High School)"
        elif drifted_value > t['difficult']:
            return "Difficult (College)"
        else:
            return "Very Difficult (Specialist)"
    else:  # Normal scale: higher is harder
        if drifted_value < t['very_easy']:
            return "Very Easy (Elementary)"
        elif drifted_value < t['easy']:
            return "Easy (Middle School)"
        elif drifted_value < t['moderate']:
            return "Moderate (High School)"
        elif drifted_value < t['difficult']:
            return "Difficult (College)"
        else:
            return "Very Difficult (Specialist)"


# Weights for metrics in overall difficulty calculation
weights = {
    'flesch_kincaid_grade': 0.45,
    'flesch_reading_ease': 0.35,
    'gunning_fog': 0.1,
    'smog_index': 0.05,
    'automated_readability_index': 0.05
}

# Collect results for summary table
results = []

# Process each text
for category, text in texts.items():
    print(f"\n=== Testing Category: {category} ===")
    try:
        # Compute readability scores using textstat
        scores = {
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text)
        }
    except Exception as e:
        print(f"Error computing readability measures: {e}")
        continue

    # Print readability scores with categorizations
    print("Readability Metrics:")
    categories = []
    weighted_scores = {}
    for metric, value in scores.items():
        category_label = categorize_score(metric, value)
        categories.append(category_label)
        # Accumulate weighted scores
        weight = weights.get(metric, 0)
        if category_label not in weighted_scores:
            weighted_scores[category_label] = 0
        weighted_scores[category_label] += weight
        print(f"  {metric}: {value:.2f} ({category_label})")

    # Determine overall difficulty (weighted majority vote)
    if weighted_scores:
        majority_category = max(weighted_scores, key=weighted_scores.get)
        print(f"Overall Difficulty: {majority_category}")
    else:
        majority_category = "N/A"
        print("No readability metrics available.")

    # Store results for summary table
    result = {'Category': category}
    for metric, value in scores.items():
        result[metric] = value
        result[f"{metric}_category"] = categorize_score(metric, value)
    result['Overall Difficulty'] = majority_category
    results.append(result)

# Create and display summary table
df = pd.DataFrame(results)
print("\n=== Summary Table ===")
print(df[['Category', 'flesch_kincaid_grade', 'flesch_kincaid_grade_category',
         'flesch_reading_ease', 'flesch_reading_ease_category',
          'gunning_fog', 'gunning_fog_category',
          'smog_index', 'smog_index_category',
          'automated_readability_index', 'automated_readability_index_category',
          'Overall Difficulty']])

# Export summary table to CSV
df.to_csv('readability_results.csv', index=False)
print("\nSummary table exported to 'readability_results.csv'")

# Create bar chart visualization
metrics = ['flesch_kincaid_grade', 'flesch_reading_ease',
           'gunning_fog', 'smog_index', 'automated_readability_index']
categories = list(texts.keys())
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.15
x = range(len(categories))

for i, metric in enumerate(metrics):
    values = [result[metric] for result in results]
    ax.bar([xi + i * bar_width for xi in x], values, bar_width, label=metric)

ax.set_xlabel('Text Category')
ax.set_ylabel('Score')
ax.set_title('Readability Scores by Category')
ax.set_xticks([xi + bar_width * (len(metrics) - 1) / 2 for xi in x])
ax.set_xticklabels(categories, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()
