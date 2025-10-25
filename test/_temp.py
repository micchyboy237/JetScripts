from jet.logger import logger
from jet.transformers.formatters import format_json
import stanza
import os
import shutil
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Fixed: Include all required processors for depparse
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

def extract_core_relations(doc):
    """Extract key dependency triples for RAG context."""
    relations = []
    doc = nlp(doc)  # Process the document here
    for sent in doc.sentences:
        for word in sent.words:
            if word.deprel in ['nsubj', 'obj', 'root']:
                head = sent.words[word.head - 1].text if word.head > 0 else "ROOT"
                relations.append((word.text, word.deprel, head))
    return relations

# Usage: filter RAG chunks by relation density/importance
contexts = extract_core_relations("Apple's CEO Tim Cook announced new AI features.")
logger.success(format_json(contexts))
save_file(contexts, f"{OUTPUT_DIR}/contexts.json")