import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# Load a blank English model
nlp = spacy.blank("en")

# Define the NER component
ner = nlp.add_pipe("ner")

# Add custom labels
labels = ["PROGRAMMING_LANGUAGE", "FRAMEWORK", "LIBRARY", "TOOL"]
for label in labels:
    ner.add_label(label)

# Example training data
TRAIN_DATA = [
    ("Python is a popular programming language.", {
     "entities": [(0, 6, "PROGRAMMING_LANGUAGE")]}),
    ("Django is a high-level Python framework.",
     {"entities": [(0, 6, "FRAMEWORK")]}),
    ("React is a JavaScript library for building user interfaces.",
     {"entities": [(0, 5, "LIBRARY")]}),
    ("Docker is a tool designed to make it easier to create, deploy, and run applications.", {
     "entities": [(0, 6, "TOOL")]}),
]

# Convert training data to spaCy's format
db = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    db.add(example.reference)

# Train the model
optimizer = nlp.begin_training()
for i in range(10):
    for example in db.get_examples(nlp):
        nlp.update([example], sgd=optimizer)

# Save the model
nlp.to_disk("software_ner_model")

# Load the trained model
nlp = spacy.load("software_ner_model")

# Test the model
doc = nlp("Flask is a micro web framework written in Python.")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
