import spacy
from spacy.tokens import Doc, Span
from spacy.training import Example

# Initialize spaCy pipeline with spancat
nlp = spacy.blank("en")
spancat = nlp.add_pipe("spancat", config={"spans_key": "sc", "threshold": 0.5})
spancat.add_label("CONDITION")

# Example training data
text = "Patient has septic shock."
doc = nlp.make_doc(text)
spans = [Span(doc, 2, 4, label="CONDITION")]
doc.spans["sc"] = spans
example = Example.from_dict(doc, {"spans": {"sc": [(2, 4, "CONDITION")]}})

# Train the model
optimizer = nlp.begin_training()
nlp.update([example], sgd=optimizer)
