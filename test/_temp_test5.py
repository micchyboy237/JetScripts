# import spacy
# from spacy import displacy
# from spacy.tokens import Span

# text = "Welcome to the Bank of China."

# nlp = spacy.blank("en")
# doc = nlp(text)

# doc.spans["sc"] = [
#     Span(doc, 3, 6, "ORG"),
#     Span(doc, 5, 6, "GPE"),
# ]

# displacy.serve(doc, style="span", port=5001)


import webbrowser
import spacy
from spacy import displacy

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
webbrowser.open('http://localhost:5001')
displacy.serve(doc, style="ent", port=5001)
