import os
import shutil
from jet._token.token_utils import token_counter
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.file.utils import save_file
from span_marker import SpanMarkerModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Load the pre-trained model
model = SpanMarkerModel.from_pretrained(
    # "tomaarsen/span-marker-bert-base-cross-ner",
    "tomaarsen/span-marker-roberta-large-ontonotes5",
).to("mps")

embed_model = "embeddinggemma"

texts = load_sample_data(model=embed_model)
token_counts = token_counter(texts, model=embed_model, prevent_total=True)
save_file([{
    "tokens": tokens,
    "text": text,
} for tokens, text in zip(token_counts, texts)], f"{OUTPUT_DIR}/texts.json")

save_file({
    "min": min(token_counts),
    "max": max(token_counts),
    "average": sum(token_counts) // len(token_counts),
    "total": sum(token_counts),
    # "sep": sep_token_count
}, f"{OUTPUT_DIR}/tokens.json")

# Predict entities
entities_list = model.predict(texts, batch_size=32, show_progress_bar=True)

save_file([{
    "index": idx,
    "text": text,
    "ent_count": len(entities),
    "entities": entities,
} for idx, (text, entities) in enumerate(zip(texts, entities_list))], f"{OUTPUT_DIR}/entities.json")
