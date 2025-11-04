from jet.code.extraction import extract_sentences
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

texts = ["Barack Obama was born in Hawaii.", "He was elected president in 2008.", "Obama attended Harvard."]

def main():
    sentences = extract_sentences(texts, use_gpu=True)
    save_file(sentences, f"{OUTPUT_DIR}/sentences.json")

if __name__ == "__main__":
    main()
