import os
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
from span_marker import SpanMarkerModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the pre-trained model
model = SpanMarkerModel.from_pretrained(
    # "tomaarsen/span-marker-bert-base-cross-ner",
    "tomaarsen/span-marker-roberta-large-ontonotes5",
).to("mps")

# Example text
texts = load_sample_data()

# Predict entities
entities = model.predict(texts, batch_size=32, show_progress_bar=True)

output_dir = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
save_file(entities, f"{output_dir}/entities.json")
