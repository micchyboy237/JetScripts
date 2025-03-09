import spacy

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")

text = "React.js and JavaScript are used in web development."
doc = nlp(text)

for token in doc:
    print(token.text, "->", token.lemma_)
