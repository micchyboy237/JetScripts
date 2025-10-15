from jet.file.utils import save_file
from jet.transformers.object import make_serializable
from stanza.server import CoreNLPClient
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

# example text
logger.gray('---')
logger.info('input text')
logger.newline()

text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."

logger.debug(text)

# set up the client
logger.gray('---')
logger.info('starting up Java Stanford CoreNLP Server...')

# set up the client
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'], timeout=60000, memory='16G', endpoint="http://localhost:9000") as client:
    # submit the request to the server
    ann = client.annotate(text)

    # get the first sentence
    sentence = ann.sentence[0]

    # get the dependency parse of the first sentence
    logger.gray('---')
    logger.info('dependency parse of first sentence')
    dependency_parse = sentence.basicDependencies
    logger.success(dependency_parse)
    save_file(make_serializable(dependency_parse), f"{OUTPUT_DIR}/dependency_parse.json")
 
    # get the constituency parse of the first sentence
    logger.gray('---')
    logger.info('constituency parse of first sentence')
    constituency_parse = sentence.parseTree
    logger.success(constituency_parse)
    save_file(make_serializable(constituency_parse), f"{OUTPUT_DIR}/constituency_parse.json")

    # get the first subtree of the constituency parse
    logger.gray('---')
    logger.info('first subtree of constituency parse')
    first_subtree = constituency_parse.child[0]
    logger.success(first_subtree)
    save_file(make_serializable(first_subtree), f"{OUTPUT_DIR}/first_subtree.json")

    # get the value of the first subtree
    logger.gray('---')
    logger.info('value of first subtree of constituency parse')
    subtree_value = constituency_parse.child[0].value
    logger.success(subtree_value)
    save_file({"value": subtree_value}, f"{OUTPUT_DIR}/subtree_value.json")

    # get the first token of the first sentence
    logger.gray('---')
    logger.info('first token of first sentence')
    token = sentence.token[0]
    logger.success(token)
    save_file(make_serializable(token), f"{OUTPUT_DIR}/first_token.json")

    # get the part-of-speech tag
    logger.gray('---')
    logger.info('part of speech tag of token')
    pos_tag = token.pos
    logger.success(pos_tag)
    save_file({"pos": pos_tag}, f"{OUTPUT_DIR}/pos_tag.json")

    # get the named entity tag
    logger.gray('---')
    logger.info('named entity tag of token')
    ner_tag = token.ner
    logger.success(ner_tag)
    save_file({"ner": ner_tag}, f"{OUTPUT_DIR}/ner_tag.json")

    # get an entity mention from the first sentence
    logger.gray('---')
    logger.info('first entity mention in sentence')
    entity_mention = sentence.mentions[0]
    logger.success(entity_mention)
    save_file(make_serializable(entity_mention), f"{OUTPUT_DIR}/first_entity_mention.json")

    # access the coref chain
    logger.gray('---')
    logger.info('coref chains for the example')
    coref_chains = ann.corefChain
    logger.success(coref_chains)
    save_file(make_serializable(coref_chains), f"{OUTPUT_DIR}/coref_chains.json")

    # Use tokensregex patterns to find who wrote a sentence.
    logger.gray('---')
    logger.info('tokensregex matches')
    pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
    matches = client.tokensregex(text, pattern)
    logger.success(matches)
    save_file(matches, f"{OUTPUT_DIR}/tokensregex_matches.json")
    # sentences contains a list with matches for each sentence.
    assert len(matches["sentences"]) == 3
    # length tells you whether or not there are any matches in this
    assert matches["sentences"][1]["length"] == 1
    # You can access matches like most regex groups.
    assert matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
    assert matches["sentences"][1]["0"]["1"]["text"] == "Chris"

    # Use semgrex patterns to directly find who wrote what.
    logger.gray('---')
    logger.info('semgrex matches')
    pattern = '{word:wrote} >nsubj {}=subject >obj {}=object'
    matches = client.semgrex(text, pattern)
    logger.success(matches)
    save_file(matches, f"{OUTPUT_DIR}/semgrex_matches.json")
    # sentences contains a list with matches for each sentence.
    assert len(matches["sentences"]) == 3
    # length tells you whether or not there are any matches in this
    assert matches["sentences"][1]["length"] == 1
    # You can access matches like most regex groups.
    assert matches["sentences"][1]["0"]["text"] == "wrote"
    assert matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
    assert matches["sentences"][1]["0"]["$object"]["text"] == "sentence"

logger.info("\n\nDONE!")