from gliner import GLiNER
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger
import torch
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load a pretrained GLiNER model (CPU-friendly)
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1", map_location=device)

# When you don't know the labels — use a generic placeholder
generic_labels = ["entity"]  # open ontology mode

texts = load_sample_data()

# Predict entities
entities_lists = model.run(texts, generic_labels, threshold=0.1, multi_label=True)
entities = [ent for ents in entities_lists for ent in ents]

logger.gray(f"RESULT ({len(entities)}):")
logger.success(format_json(entities))

save_file(entities, f"{OUTPUT_DIR}/entities.json")
