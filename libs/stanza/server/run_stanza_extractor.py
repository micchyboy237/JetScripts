# JetScripts/libs/stanza/server/run_stanza_extractor.py
from typing import Dict, Any, List
from jet.libs.stanza.server.stanza_extractor import StanzaExtractor
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

TEXT = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

def example_annotate() -> Dict[str, Any]:
    """Given text, annotate with full NLP pipeline."""
    extractor = StanzaExtractor()
    extractor.start()
    ann = extractor.annotate_text(TEXT)
    plain = extractor.get_plain_text(ann)
    extractor.stop()
    print("\n[ANNOTATED TEXT]\n", plain)
    return ann

def example_tokensregex() -> Dict[str, Any]:
    """Example using TokensRegex pattern."""
    extractor = StanzaExtractor()
    extractor.start()
    pattern = "([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/"
    result = extractor.tokensregex(TEXT, pattern)
    extractor.stop()
    print("\n[TOKENSREGEX MATCHES]\n", result)
    return result

def example_semgrex() -> List[Dict[str, Any]]:
    """Example using Semgrex dependency pattern."""
    extractor = StanzaExtractor()
    extractor.start()
    pattern = "{word:wrote} >nsubj {}=subject >obj {}=object"
    matches = extractor.semgrex(TEXT, pattern, to_words=True)
    extractor.stop()
    print("\n[SEMGREX MATCHES]\n", matches)
    return matches

def example_tregex() -> Dict[str, Any]:
    """Example using Tregex phrase-structure pattern."""
    extractor = StanzaExtractor()
    extractor.start()
    pattern = "PP < NP"
    result = extractor.tregex(TEXT, pattern)
    extractor.stop()
    print("\n[TREGEX MATCHES]\n", result)
    return result

def example_tregex_on_trees() -> Dict[str, Any]:
    """Example applying Tregex to explicit parse trees."""
    extractor = StanzaExtractor()
    extractor.start()
    trees = extractor.parse_trees(
        "(ROOT (S (NP (NNP Jennifer)) (VP (VBZ has) (NP (JJ blue) (NN skin)))))"
        "(ROOT (S (NP (PRP I)) (VP (VBP like) (NP (PRP$ her) (NNS antennae)))))"
    )
    pattern = "VP < NP"
    matches = extractor.tregex(pattern, trees=trees)
    extractor.stop()
    print("\n[TREGEX TREES MATCHES]\n", matches)
    return matches

if __name__ == "__main__":
    logger.info("Running example_annotate()...")
    results = example_annotate()
    save_file(results, f"{OUTPUT_DIR}/annotate_results.json")
    
    logger.info("Running example_tokensregex()...")
    results = example_tokensregex()
    save_file(results, f"{OUTPUT_DIR}/tokensregex_results.json")
    
    logger.info("Running example_semgrex()...")
    results = example_semgrex()
    save_file(results, f"{OUTPUT_DIR}/semgrex_results.json")
    
    logger.info("Running example_tregex()...")
    results = example_tregex()
    save_file(results, f"{OUTPUT_DIR}/tregex_results.json")
    
    logger.info("Running example_tregex_on_trees()...")
    results = example_tregex_on_trees()
    save_file(results, f"{OUTPUT_DIR}/tregex_on_trees_results.json")
