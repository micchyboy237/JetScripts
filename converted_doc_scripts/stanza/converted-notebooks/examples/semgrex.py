import stanza
from stanza.server.semgrex import Semgrex

from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

doc = nlp("Banning opal removed all artifact decks from the meta.  I miss playing lantern.")
with Semgrex(classpath="$CLASSPATH") as sem:
    semgrex_results = sem.process(doc,
                                  "{pos:NN}=object <obl {}=action",
                                  "{cpos:NOUN}=thing <obj {cpos:VERB}=action")
    logger.newline()
    logger.gray("COMPLETE RESULTS")
    logger.success(semgrex_results)

    logger.newline()
    logger.gray("Number of matches in graph 0 ('Banning opal...') for semgrex query 1 (thing <obj action): %d" % len(semgrex_results.result[0].result[1].match))
    for match_idx, match in enumerate(semgrex_results.result[0].result[1].match):
        logger.success("Match {}:\n-----------\n{}".format(match_idx + 1, match))

    logger.teal("graph 1 for semgrex query 0 is an empty match: len %d" % len(semgrex_results.result[1].result[0].match))
