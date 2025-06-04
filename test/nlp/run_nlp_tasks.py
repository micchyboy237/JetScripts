import spacy
from spacy.training import Example


def print_pipeline_names(nlp):
    print("\n[log] Retrieving pipeline component names...")
    try:
        pipe_names = nlp.pipe_names
        print("[pipeline_names]")
        if pipe_names:
            for name in pipe_names:
                print(f"Component: {name}")
        else:
            print("No components in the pipeline.")
        return pipe_names
    except AttributeError as e:
        print("[pipeline_names]")
        print(f"Error accessing pipeline: {e}")
        return []


def use_tok2vec(text: str, nlp):
    print("\n[log] Processing text to access tok2vec vectors...")
    doc = nlp(text)
    print("[tok2vec]")
    for token in doc:
        print(
            f"Token: {token.text}, Vector (first 5 dims): {token.vector[:5]}")


def use_tagger(text: str, nlp):
    print("\n[log] Processing text for POS tagging...")
    doc = nlp(text)
    print("[tagger]")
    for token in doc:
        print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}")


def use_senter(text: str, nlp):
    print("\n[log] Processing text for sentence segmentation...")
    doc = nlp(text)
    print("[senter]")
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")


def use_parser(text: str, nlp):
    print("\n[log] Processing text for dependency parsing...")
    doc = nlp(text)
    print("[parser]")
    for token in doc:
        print(
            f"Token: {token.text}, Dep: {token.dep_}, Head: {token.head.text}")


def use_lemmatizer(text: str, nlp):
    print("\n[log] Processing text for lemmatization...")
    doc = nlp(text)
    print("[lemmatizer]")
    for token in doc:
        print(f"Token: {token.text}, Lemma: {token.lemma_}")


def use_attribute_ruler(text: str, nlp):
    print("\n[log] Adding custom pattern to attribute_ruler...")
    ruler = nlp.get_pipe("attribute_ruler")
    ruler.add(patterns=[[{"TEXT": "spaCy"}]],
              attrs={"POS": "PROPN", "TAG": "NNP"})
    print("\n[log] Processing text with attribute_ruler...")
    doc = nlp(text)
    print("[attribute_ruler]")
    for token in doc:
        print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}")


def use_ner(text: str, nlp):
    print("\n[log] Processing text for named entity recognition...")
    doc = nlp(text)
    print("[ner]")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")


def use_sentencizer(text: str, nlp):
    print("\n[log] Processing text for rule-based sentence segmentation...")
    doc = nlp(text)
    print("[sentencizer]")
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")


def use_span_marker(text: str, nlp):
    print("\n[log] Processing text with SpanMarker...")
    try:
        doc = nlp(text)
        print("[span_marker]")
        if doc.ents:
            for ent in doc.ents:
                print(f"Entity: {ent.text}, Label: {ent.label_}")
        else:
            print("No entities detected.")
    except Exception as e:
        print("[span_marker]")
        print(f"SpanMarker model failed to load or process: {e}")


def use_textcat(text: str, nlp):
    print("\n[log] Processing text for sentiment analysis...")
    doc = nlp(text)
    print("[textcat]")
    if doc.cats:
        for label, score in doc.cats.items():
            print(f"Label: {label}, Score: {score:.4f}")
    else:
        print("No sentiment scores available. Ensure textcat is trained.")


def use_spancat(text: str, nlp):
    print("\n[log] Processing text for span categorization...")
    doc = nlp(text)
    print("[spancat]")
    if doc.spans.get("sc", []):
        for span in doc.spans["sc"]:
            print(f"Span: {span.text}, Label: {span.label_}")
    else:
        print("No spans detected. Ensure spancat is trained.")


def train_custom_pipelines(text: str, nlp):
    # Add 'textcat' component if not present
    if "textcat" not in nlp.pipe_names:
        print("\n[log] Adding 'textcat' component...")
        textcat = nlp.add_pipe("textcat", last=True)
        textcat.add_label("POSITIVE")
        textcat.add_label("NEGATIVE")
        textcat.add_label("NEUTRAL")

    # Add 'spancat' component if not present WITHOUT 'labels' in config
    if "spancat" not in nlp.pipe_names:
        print("\n[log] Adding 'spancat' component...")
        spancat = nlp.add_pipe("spancat", last=True,
                               config={"spans_key": "sc"})
        spancat.add_label("ORGANIZATION")

    print("\n[log] Initializing optimizer...")
    optimizer = nlp.create_optimizer()

    print("\n[log] Training 'textcat' and 'spancat' components...")

    train_data = [
        (text, {
            "cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0},
            "spans": {"sc": [(35, 46, "ORGANIZATION")]}
        })
    ]

    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)


def main():
    sample_text = "spaCy is an NLP library created by Explosion AI. It is used in many applications."

    print("\n[log] Loading 'en_core_web_sm' pipeline...")
    nlp = spacy.load("en_core_web_sm")

    # Load auxiliary pipelines for sentence segmentation
    nlp_senter = spacy.load("en_core_web_sm", disable=["parser"])
    if "sentencizer" not in nlp_senter.pipe_names:
        nlp_senter.add_pipe("sentencizer")

    nlp_sentencizer = spacy.blank("en")
    nlp_sentencizer.add_pipe("sentencizer")

    # Attempt to load span_marker pipeline if available
    try:
        nlp_span_marker = spacy.load("en_core_web_sm", exclude=["ner"])
        nlp_span_marker.add_pipe("span_marker", config={
            "model": "tomaarsen/span-marker-mbert-base-multinerd"
        })
    except Exception as e:
        print(f"[log] Failed to load SpanMarker: {e}")
        nlp_span_marker = None

    # Run pipeline inspection and component demos
    print_pipeline_names(nlp)
    use_tok2vec(sample_text, nlp)
    use_tagger(sample_text, nlp)
    use_parser(sample_text, nlp)
    use_lemmatizer(sample_text, nlp)
    use_attribute_ruler(sample_text, nlp)
    use_ner(sample_text, nlp)
    use_senter(sample_text, nlp_senter)
    use_sentencizer(sample_text, nlp_sentencizer)
    if nlp_span_marker:
        use_span_marker(sample_text, nlp_span_marker)

    # Train custom pipelines
    # train_custom_pipelines(sample_text, nlp)
    # use_textcat(sample_text, nlp)
    # use_spancat(sample_text, nlp)


if __name__ == "__main__":
    main()
