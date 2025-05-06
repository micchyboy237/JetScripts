import shutil
import os
from jet.file.utils import load_file, save_file
from jet.wordnet.analyzers.analyze_ngrams import analyze_ngrams


def main_analyze_ngrams(texts, texts_dict, output_dir):
    filtered_data = analyze_ngrams(texts, texts_dict, min_tfidf=0.03)
    if not filtered_data:
        print("No texts selected. Saving empty output.")
        filtered_data = []
    os.makedirs(output_dir, exist_ok=True)
    save_file(filtered_data, os.path.join(output_dir, 'ngrams.json'))


if __name__ == '__main__':
    texts = [
        "This smartphone has an excellent battery life and a very fast processor. Highly recommend it for gaming.",
        "The camera quality is outstanding, but the battery drains quickly when recording videos.",
        "I love the sleek design and vibrant display. The phone is worth every penny.",
        "The processor is fast, but the software updates are slow and buggy. Disappointed with the support.",
        "Amazing battery life and a fantastic camera. This phone exceeded my expectations.",
        "The display is stunning, but the phone overheats during heavy use. Not ideal for multitasking.",
        "Great value for money. The camera and processor are top-notch, but the battery could be better.",
        "The phone has a premium feel, but the customer service was unhelpful when I had issues.",
        "Fantastic gaming performance with a smooth display. Battery life is decent but not the best.",
        "The camera struggles in low light, and the processor lags during heavy apps. Very frustrating.",
        "The battery lasts all day, and the camera is crystal clear. Great phone!",
        "Software glitches make this phone frustrating despite a good processor.",
        "The display is gorgeous, but the battery life is mediocre at best."
    ]

    texts_dict = {
        texts[i]: {"id": i+1, "rating": [5, 4, 5, 3, 5, 3, 4,
                                         3, 4, 2, 4, 3, 3][i], "category": "smartphone"}
        for i in range(len(texts))
    }

    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    main_analyze_ngrams(texts, texts_dict, output_dir)
