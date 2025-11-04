from jet.libs.gliner_spacy.gliner_pipeline_utils import (
    CategoryData,
    init_gliner_pipeline,
    load_text_from_file,
    process_text,
    extract_sentence_themes,
    visualize_doc,
)

# --- Define category dataset ---
cat_data: CategoryData = {
    "family": ["child", "spouse", "family", "parent"],
    "labor": ["work", "job", "office"],
    "education": ["school", "student"],
    "movement": ["verb of movement", "place of movement"],
    "violence": ["violence", "weapon", "attack", "fear"],
}

# --- Initialize pipeline ---
nlp = init_gliner_pipeline(cat_data)

# --- Load sample text ---
text = load_text_from_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/gliner_spacy/examples/gliner_cat/data/testimony.txt")

# --- Run inference ---
doc = process_text(nlp, text)

# --- Inspect sentence 200 ---
result = extract_sentence_themes(doc, 200)
print(result)

# --- Visualize ---
visualize_doc(doc, sent_start=0, sent_end=300, chunk_size=2, fig_h=10)
