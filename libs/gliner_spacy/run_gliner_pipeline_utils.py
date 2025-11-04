from jet.libs.gliner_spacy.gliner_pipeline_utils import (
    CategoryData,
    init_gliner_pipeline,
    process_text,
    extract_sentence_themes,
    visualize_doc,
)
from jet._token.token_utils import token_counter
from jet.libs.bertopic.examples.mock import load_sample_data
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

embed_model = "embeddinggemma"
chunks = load_sample_data(model=embed_model)
token_counts = token_counter(chunks, model=embed_model, prevent_total=True)

save_file([
    {
        "chunk_idx": idx,
        "tokens": tokens,
        "chunk": chunk,
    }
    for idx, (tokens, chunk) in enumerate(zip(token_counts, chunks))
], f"{OUTPUT_DIR}/chunks.json")

save_file({
    "min": min(token_counts),
    "ave": sum(token_counts) // len(token_counts),
    "max": max(token_counts),
    "total": sum(token_counts),
}, f"{OUTPUT_DIR}/tokens.json")

for chunk_idx, chunk in enumerate(chunks):
    sub_output_dir = f"{OUTPUT_DIR}/chunk_{chunk_idx + 1}"
    save_file(chunk, f"{sub_output_dir}/chunk.txt")

    logger.info("--- Run inference ---")
    doc = process_text(nlp, chunk)

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
