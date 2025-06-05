import os
from jet.file.utils import save_file
from span_marker import SpanMarkerModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the pre-trained model
model = SpanMarkerModel.from_pretrained(
    # "tomaarsen/span-marker-bert-base-cross-ner",
    "tomaarsen/span-marker-roberta-large-ontonotes5",
)

# Example text
text = "Explore Exciting Opportunities at Vault Outsourcing: Your Gateway to Offshoring Excellence:\n\nAre you seeking a great career opportunity with exceptional benefits? Look no further! Vault Outsourcing is not just a company; it's a dynamic force offering a new and exciting career path.\nWe believe our people are our greatest asset and foster a family atmosphere that encourages excellence. Join us in redefining offshoring excellence, where your career is valued, and exciting opportunities await. Discover what we can offer - your gateway to a fulfilling career!"

# Predict entities
entities = model.predict(text)

print(entities)


output_dir = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
save_file(entities, f"{output_dir}/entities.json")
