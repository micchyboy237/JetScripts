import os
import shutil
from jet.file.utils import save_file
from jet.logger import logger
import pandas as pd
import matplotlib.pyplot as plt
from jet.wordnet.analyzers.text_analysis import analyze_readability

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
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

    # Collect results for summary table
    results = []

    # Process each text
    for category, text in texts.items():
        print(f"\n=== Testing Category: {category} ===")
        try:
            # Compute readability scores using analyze_readability
            result = analyze_readability(text)
            scores = result['scores']
            categories = result['categories']
            overall_difficulty = result['overall_difficulty_description']
        except Exception as e:
            print(f"Error computing readability measures: {e}")
            continue

        # Print readability scores with categorizations
        print("Readability Metrics:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.2f} ({categories[metric]})")
        print(f"Overall Difficulty: {overall_difficulty}")

        # Store results for summary table
        result_entry = {
            'Category': category,
            'Text': text.strip()  # Store the original text
        }
        for metric, value in scores.items():
            result_entry[metric] = value
            result_entry[f"{metric}_category"] = categories[metric]
        result_entry['Overall Difficulty'] = overall_difficulty
        results.append(result_entry)

    results_file = f"{output_dir}/results.json"
    save_file(results, results_file)

    # Create and display summary table
    df = pd.DataFrame(results)
    print("\n=== Summary Table ===")
    # Exclude 'Text' from display to keep table clean
    display_columns = [
        'Category',
        'flesch_kincaid_grade', 'flesch_kincaid_grade_category',
        'flesch_reading_ease', 'flesch_reading_ease_category',
        'gunning_fog', 'gunning_fog_category',
        'smog_index', 'smog_index_category',
        'automated_readability_index', 'automated_readability_index_category',
        'Overall Difficulty'
    ]
    print(df[display_columns])

    # Export summary table to CSV, including 'Text' column
    results_table_file = f"{output_dir}/results_table.csv"
    df.to_csv(results_table_file, index=False)
    logger.success(f"\nSummary table exported to {results_table_file}")

    # Create bar chart visualization
    metrics = ['flesch_kincaid_grade', 'flesch_reading_ease',
               'gunning_fog', 'smog_index', 'automated_readability_index']
    categories = list(texts.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.15
    x = range(len(categories))

    for i, metric in enumerate(metrics):
        values = [result[metric] for result in results]
        ax.bar([xi + i * bar_width for xi in x],
               values, bar_width, label=metric)

    ax.set_xlabel('Text Category')
    ax.set_ylabel('Score')
    ax.set_title('Readability Scores by Category')
    ax.set_xticks([xi + bar_width * (len(metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    plt.tight_layout()
    chart_file = f"{output_dir}/readability_chart.png"
    plt.savefig(chart_file)
    logger.success(f"\nAnalysis chart exported to {chart_file}")
