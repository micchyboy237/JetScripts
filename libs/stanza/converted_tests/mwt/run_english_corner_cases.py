import stanza
from stanza.tests import TEST_MODELS_DIR
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil
import json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def example_mwt_unknown_char():
    """Example: Demonstrate English MWT behavior for unknown characters"""
    pipeline = stanza.Pipeline(processors='tokenize,mwt', dir=TEST_MODELS_DIR, lang='en', download_method=None)
    mwt_trainer = pipeline.processors['mwt']._trainer
    
    results = {
        "function": "mwt_unknown_char",
        "tests": []
    }
    
    # Use a non-standard 'i' letter to test MWT robustness
    for letter in "ĩîíìī":
        if letter not in mwt_trainer.vocab:
            word = "Jenn" + letter + "fer"
            possessive = word + "'s"
            text = f"I wanna lick {possessive} antennae"
            doc = pipeline(text)
            
            test_result = {
                "input_text": text,
                "tokens": [
                    {
                        "token_text": token.text,
                        "words": [w.text for w in token.words]
                    } for token in doc.sentences[0].tokens
                ]
            }
            results["tests"].append(test_result)
            
            logger.debug(f"Input text: {text}")
            logger.gray("Tokens:")
            for token in doc.sentences[0].tokens:
                logger.success(f"  {token.text:>15} -> {[w.text for w in token.words]}")
            return results
            
    results["message"] = "All variant 'i' letters are in the vocab; no unknown char found."
    logger.success(results["message"])
    return results

def example_english_mwt_casing():
    """Example: Demonstrate correct casing in English MWT expansion"""
    pipeline = stanza.Pipeline(processors='tokenize,mwt', dir=TEST_MODELS_DIR, lang='en', download_method=None)
    
    results = {
        "function": "english_mwt_casing",
        "tests": []
    }
    
    examples = [
        "SHE'S GOT NICE ANTENNAE",
        "She's GOT NICE ANTENNAE",
        "JENNIFER HAS NICE ANTENNAE"
    ]
    
    for text in examples:
        doc = pipeline(text)
        test_result = {
            "input_text": text,
            "expanded_words": [w.text for w in doc.sentences[0].words]
        }
        results["tests"].append(test_result)
        
        logger.debug(f"\nInput: {text}")
        logger.success("Expanded words:", test_result["expanded_words"])
    
    return results

if __name__ == "__main__":
    logger.info("=== Example 1: Unknown Character Handling ===")
    results1 = example_mwt_unknown_char()
    save_file(json.dumps(results1, indent=2), f"{OUTPUT_DIR}/mwt_unknown_char_results.json")

    logger.info("\n=== Example 2: English Casing Behavior ===")
    results2 = example_english_mwt_casing()
    save_file(json.dumps(results2, indent=2), f"{OUTPUT_DIR}/english_mwt_casing_results.json")
