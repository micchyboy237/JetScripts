import os
import shutil
from jet.file.utils import save_file
from jet.wordnet.analyzers.text_analysis import analyze_text

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    # Sample text for analysis
    text = """
    The quick brown fox jumps over the lazy dog. This sentence is simple yet effective.
    Reading comprehension is vital for learning. Let's analyze this text thoroughly.
    Some words are complex, while others are straightforward. This diversity aids analysis.
    """

    # Analyze text and print results
    stats = analyze_text(text)
    for metric, value in stats.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")
    save_file({"text": text, "stats": stats}, f"{output_dir}/stats.json")
