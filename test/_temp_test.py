import readability
from collections import Counter

# Sample texts for each category (unchanged)
texts = {
    "Very Easy (Elementary)": """
The zoo is fun. I see big lions. They roar loud. Monkeys swing on trees. They eat bananas. The elephant is huge. It sprays water with its trunk. I like the giraffe. It has a long neck. It eats leaves from tall trees. My mom buys me ice cream. We walk and laugh. The zoo is the best place.
""",
    "Easy (Middle School)": """
My favorite hobby is drawing. I use pencils and crayons. I draw animals and trees. Sometimes, I make pictures of my family. Drawing makes me happy. I practice every day after school. My teacher says my art is good. Last week, I drew a big dog. It had fluffy fur and a wagging tail. I showed it to my friends. They liked it. I want to be an artist when I grow up. Drawing is fun and relaxing.
""",
    "Moderate (High School)": """
The American Revolution changed history. It started in 1775. Colonists in the Thirteen Colonies wanted freedom from British rule. Taxes, like the Stamp Act, made them angry. In 1776, they wrote the Declaration of Independence. This document explained their reasons for rebellion. Leaders like George Washington fought bravely. Battles at Lexington and Yorktown were important. By 1783, the Treaty of Paris ended the war. America became a new nation. The revolution inspired other countries to seek freedom.
""",
    "Difficult (College)": """
Climate change poses significant challenges to global ecosystems. Rising temperatures, driven by greenhouse gas emissions, accelerate polar ice melt and sea-level rise. Coastal communities face increased flooding risks, while agricultural yields decline due to unpredictable weather patterns. Biodiversity loss threatens species survival, particularly in tropical regions. Mitigation strategies, such as renewable energy adoption and carbon pricing, are critical. However, international cooperation remains inconsistent, hindering progress. Scientific consensus underscores the urgency of addressing these issues to prevent irreversible environmental damage. Policy reform and public awareness are essential for sustainable outcomes.
""",
    "Very Difficult (Specialist)": """
Quantum computing represents a paradigm shift in computational theory. Unlike classical systems, quantum computers leverage superposition and entanglement to perform calculations at unprecedented speeds. Qubits, the fundamental units, exist in multiple states simultaneously, enabling exponential scaling of processing power. Recent advancements in error correction and coherence times have mitigated decoherence challenges, enhancing system stability. Applications in cryptography, optimization, and molecular simulation are transformative, though scalability remains a barrier. Interdisciplinary collaboration between physicists, engineers, and computer scientists is essential to overcome technical limitations and realize quantum supremacy in practical contexts.
"""
}

# Define thresholds for readability metrics (unchanged)
thresholds = {
    'Kincaid': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 13},
    'ARI': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 13},
    'Coleman-Liau': {'very_easy': 6, 'easy': 9, 'moderate': 12, 'difficult': 15},
    'FleschReadingEase': {'very_easy': 80, 'easy': 60, 'moderate': 40, 'difficult': 20},
    'GunningFogIndex': {'very_easy': 6, 'easy': 9, 'moderate': 12, 'difficult': 15},
    'LIX': {'very_easy': 30, 'easy': 45, 'moderate': 60, 'difficult': 75},
    'SMOGIndex': {'very_easy': 6, 'easy': 9, 'moderate': 12, 'difficult': 15},
    'RIX': {'very_easy': 2, 'easy': 4, 'moderate': 6, 'difficult': 8},
    'DaleChallIndex': {'very_easy': 5, 'easy': 6.5, 'moderate': 8, 'difficult': 9.5}
}

# Function to categorize a readability score (unchanged)


def categorize_score(metric, value):
    if metric not in thresholds:
        return "N/A"
    t = thresholds[metric]
    if metric == 'FleschReadingEase':  # Inverted scale: higher is easier
        if value > t['very_easy']:
            return "Very Easy (Elementary)"
        elif value > t['easy']:
            return "Easy (Middle School)"
        elif value > t['moderate']:
            return "Moderate (High School)"
        elif value > t['difficult']:
            return "Difficult (College)"
        else:
            return "Very Difficult (Specialist)"
    else:  # Normal scale: higher is harder
        if value < t['very_easy']:
            return "Very Easy (Elementary)"
        elif value < t['easy']:
            return "Easy (Middle School)"
        elif value < t['moderate']:
            return "Moderate (High School)"
        elif value < t['difficult']:
            return "Difficult (College)"
        else:
            return "Very Difficult (Specialist)"


# Process each text
for category, text in texts.items():
    print(f"\n=== Testing Category: {category} ===")
    try:
        scores = readability.getmeasures(text, lang='en')
    except Exception as e:
        print(f"Error computing readability measures: {e}")
        continue

    # Process all readability measures
    categories = []
    for measure_category, metrics in scores.items():
        print(f"\nCategory: {measure_category}")
        for metric, value in metrics.items():
            if measure_category == 'readability grades':
                category_label = categorize_score(metric, value)
                categories.append(category_label)
                print(f"  {metric}: {value:.2f} ({category_label})")
            else:
                print(f"  {metric}: {value:.2f}")

    # Determine overall difficulty for readability grades
    if categories:
        majority_category = Counter(categories).most_common(1)[0][0]
        print(
            f"\nOverall Difficulty (Readability Grades): {majority_category}")
    else:
        print("\nNo readability grades available.")
