from jet.file.utils import load_file
from jet.libs.txtai.pipeline.segmentation import Segmentation
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.logger import logger
from jet.wordnet.sentence import split_sentences, adaptive_split


segmenter = Segmentation(paragraphs=True)
text = "This is the first paragraph. It contains multiple sentences to demonstrate segmentation.\n\nThis is the second paragraph, which should be recognized as a separate entity by the segmentation module."
expected = [
    "This is the first paragraph. It contains multiple sentences to demonstrate segmentation.",
    "This is the second paragraph, which should be recognized as a separate entity by the segmentation module."
]
result = segmenter(text)


data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
data = load_file(data_file)

docs = []
for text in data:
    segment = segmenter(text)
    splitted1 = split_sentences(text)
    splitted2 = adaptive_split(text, max_tokens=100)
    docs.append(segment)

copy_to_clipboard(docs)
