from jet.file.utils import save_file
from jet.transformers.object import make_serializable
from stanza.server import CoreNLPClient
from jet.logger import logger
import os
import shutil
from typing import Any, Dict, List
from tqdm import tqdm

class CoreNLPProcessor:
    """A class to manage CoreNLP client and process text annotations."""
    
    def __init__(self, annotators: List[str] = ['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'], timeout: int = 60000, memory: str = '16G', endpoint: str = "http://localhost:9000"):
        """Initialize the CoreNLPProcessor with client configuration."""
        self.client = CoreNLPClient(annotators=annotators, timeout=timeout, memory=memory, endpoint=endpoint)
        self.sentences: List[Any] = []
        self.annotation: Any = None  # Store the full annotation

    def __enter__(self):
        """Enter context manager, return self."""
        self.client.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, stop the client."""
        self.client.stop()

    def annotate_text(self, text: str) -> None:
        """Annotate text and store sentences and full annotation."""
        logger.gray('---')
        logger.info('input text')
        logger.newline()
        logger.debug(text)
        self.annotation = self.client.annotate(text)
        self.sentences = self.annotation.sentence

    def get_dependencies(self, output_dir: str) -> List[Any]:
        """Get dependency parses for all sentences."""
        logger.gray('---')
        logger.info('dependency parses for all sentences')
        dependencies = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing dependencies")):
            dep = sentence.basicDependencies
            dependencies.append(dep)
            logger.success(f"Dependency parse for sentence {i+1}: {dep}")
            save_parse_result(dep, f"{output_dir}/dependency_parse_sentence_{i+1}.json")
        return dependencies

    def get_constituency_tree(self, output_dir: str) -> List[Any]:
        """Get constituency trees for all sentences."""
        logger.gray('---')
        logger.info('constituency parses for all sentences')
        trees = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing constituency trees")):
            tree = sentence.parseTree
            trees.append(tree)
            logger.success(f"Constituency parse for sentence {i+1}: {tree}")
            save_parse_result(tree, f"{output_dir}/constituency_parse_sentence_{i+1}.json")
        return trees

    def get_subtrees(self, output_dir: str) -> List[List[Any]]:
        """Get all subtrees for constituency parses of all sentences."""
        logger.gray('---')
        logger.info('subtrees for all constituency parses')
        all_subtrees = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing subtrees")):
            tree = sentence.parseTree
            subtrees = tree.child if tree.child else []
            all_subtrees.append(subtrees)
            logger.success(f"Subtrees for sentence {i+1}: {subtrees}")
            save_parse_result(subtrees, f"{output_dir}/subtrees_sentence_{i+1}.json")
        return all_subtrees

    def get_subtree_value(self, output_dir: str) -> List[str]:
        """Get the value of the first subtree for each sentence's constituency parse."""
        logger.gray('---')
        logger.info('values of first subtrees for all sentences')
        values = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing subtree values")):
            tree = sentence.parseTree
            subtree = tree.child[0] if tree.child else None
            value = subtree.value if subtree else ""
            values.append(value)
            logger.success(f"Subtree value for sentence {i+1}: {value}")
            save_parse_result({"value": value}, f"{output_dir}/subtree_value_sentence_{i+1}.json")
        return values

    def get_tokens(self, output_dir: str) -> List[List[Any]]:
        """Get all tokens for all sentences."""
        logger.gray('---')
        logger.info('tokens for all sentences')
        all_tokens = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing tokens")):
            tokens = sentence.token if sentence.token else []
            all_tokens.append(tokens)
            logger.success(f"Tokens for sentence {i+1}: {tokens}")
            save_parse_result(tokens, f"{output_dir}/tokens_sentence_{i+1}.json")
        return all_tokens

    def get_pos_tag(self, output_dir: str) -> List[str]:
        """Get the part-of-speech tag of the first token for each sentence."""
        logger.gray('---')
        logger.info('part of speech tags for first tokens')
        pos_tags = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing POS tags")):
            token = sentence.token[0] if sentence.token else None
            pos = token.pos if token else ""
            pos_tags.append(pos)
            logger.success(f"POS tag for sentence {i+1}: {pos}")
            save_parse_result({"pos": pos}, f"{output_dir}/pos_tag_sentence_{i+1}.json")
        return pos_tags

    def get_ner_tag(self, output_dir: str) -> List[str]:
        """Get the named entity tag of the first token for each sentence."""
        logger.gray('---')
        logger.info('named entity tags for first tokens')
        ner_tags = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing NER tags")):
            token = sentence.token[0] if sentence.token else None
            ner = token.ner if token else ""
            ner_tags.append(ner)
            logger.success(f"NER tag for sentence {i+1}: {ner}")
            save_parse_result({"ner": ner}, f"{output_dir}/ner_tag_sentence_{i+1}.json")
        return ner_tags

    def get_entity_mentions(self, output_dir: str) -> List[List[Any]]:
        """Get all entity mentions for all sentences."""
        logger.gray('---')
        logger.info('entity mentions for all sentences')
        all_mentions = []
        for i, sentence in enumerate(tqdm(self.sentences, desc="Processing entity mentions")):
            mentions = sentence.mentions if sentence.mentions else []
            all_mentions.append(mentions)
            logger.success(f"Entity mentions for sentence {i+1}: {mentions}")
            save_parse_result(mentions, f"{output_dir}/entity_mentions_sentence_{i+1}.json")
        return all_mentions

    def get_coref_chains(self, output_dir: str) -> List[Any]:
        """Get coreference chains from the annotation."""
        logger.gray('---')
        logger.info('coref chains for the example')
        coref_chains = self.annotation.corefChain if self.annotation and hasattr(self.annotation, 'corefChain') else []
        logger.success(coref_chains)
        save_parse_result(coref_chains, f"{output_dir}/coref_chains.json")
        return coref_chains

    def get_tokensregex_matches(self, text: str, pattern: str, output_dir: str) -> Dict[str, Any]:
        """Get tokensregex matches for the text."""
        logger.gray('---')
        logger.info('tokensregex matches')
        matches = self.client.tokensregex(text, pattern)
        logger.success(matches)
        save_parse_result(matches, f"{output_dir}/tokensregex_matches.json")
        return matches

    def get_semgrex_matches(self, text: str, pattern: str, output_dir: str) -> Dict[str, Any]:
        """Get semgrex matches for the text."""
        logger.gray('---')
        logger.info('semgrex matches')
        matches = self.client.semgrex(text, pattern)
        logger.success(matches)
        save_parse_result(matches, f"{output_dir}/semgrex_matches.json")
        return matches

def setup_output_directory(file_path: str) -> str:
    """Set up the output directory for saving results."""
    output_dir = os.path.join(
        os.path.dirname(file_path), "generated", os.path.splitext(os.path.basename(file_path))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def initialize_logger(output_dir: str) -> str:
    """Initialize logger with a log file in the output directory."""
    log_file = os.path.join(output_dir, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    return log_file

def save_parse_result(data: Any, file_path: str) -> None:
    """Save parse result to a JSON file."""
    save_file(make_serializable(data), file_path)

def main():
    # Setup output directory and logger
    output_dir = setup_output_directory(__file__)
    initialize_logger(output_dir)

    # Example text
    text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."

    # Set up the processor
    logger.gray('---')
    logger.info('starting up Java Stanford CoreNLP Server...')
    with CoreNLPProcessor() as processor:
        processor.annotate_text(text)

        # Process all sentences
        processor.get_dependencies(output_dir)
        processor.get_constituency_tree(output_dir)
        processor.get_subtrees(output_dir)
        processor.get_subtree_value(output_dir)
        processor.get_tokens(output_dir)
        processor.get_pos_tag(output_dir)
        processor.get_ner_tag(output_dir)
        processor.get_entity_mentions(output_dir)
        processor.get_coref_chains(output_dir)

        # Tokensregex matches
        pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
        matches = processor.get_tokensregex_matches(text, pattern, output_dir)
        assert len(matches["sentences"]) == 3
        assert matches["sentences"][1]["length"] == 1
        assert matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
        assert matches["sentences"][1]["0"]["1"]["text"] == "Chris"

        # Semgrex matches
        pattern = '{word:wrote} >nsubj {}=subject >obj {}=object'
        matches = processor.get_semgrex_matches(text, pattern, output_dir)
        assert len(matches["sentences"]) == 3
        assert matches["sentences"][1]["length"] == 1
        assert matches["sentences"][1]["0"]["text"] == "wrote"
        assert matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
        assert matches["sentences"][1]["0"]["$object"]["text"] == "sentence"

    logger.info("\n\nDONE!")

if __name__ == "__main__":
    main()