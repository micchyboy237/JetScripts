from jet.libs.gliner_spacy.gliner_pipeline_utils import (
    CategoryData,
    init_gliner_pipeline,
    process_text,
    extract_sentence_themes,
    visualize_doc,
)
from jet.wordnet.text_chunker import chunk_texts
from jet.file.utils import load_file, save_file
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

logger.info("--- Define category dataset ---")
cat_data: CategoryData = {
    "family": ["child", "spouse", "family", "parent"],
    "labor": ["work", "job", "office"],
    "education": ["school", "student"],
    "movement": ["verb of movement", "place of movement"],
    "violence": ["violence", "weapon", "attack", "fear"],
}

logger.info("--- Initialize pipeline ---")
nlp = init_gliner_pipeline(cat_data)

logger.info("--- Load sample text ---")
text = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/gliner_spacy/examples/gliner_cat/data/testimony.txt")

chunks = chunk_texts(
    text,
    chunk_size=128,
    chunk_overlap=32,
    model="embeddinggemma",
    show_progress=True,
)

for chunk_idx, chunk in enumerate(chunks):
    sub_output_dir = f"{OUTPUT_DIR}/chunk_{chunk_idx + 1}"

    logger.info("--- Run inference ---")
    doc = process_text(nlp, chunk)
    save_file(doc, f"{sub_output_dir}/doc.json")

    logger.info("--- Inspect sentence 200 ---")
    sentence_themes = extract_sentence_themes(doc)
    save_file(sentence_themes, f"{sub_output_dir}/sentence_themes.json")

    logger.info("--- Visualize ---")
    image = visualize_doc(doc)
    if image:
        image_path = f"{sub_output_dir}/gliner_visualization.png"
        image.save(image_path, format="PNG")
        logger.success(f"✅ Image saved to {image_path}")
    else:
        logger.error("⚠️ No image returned by visualize_doc.")
