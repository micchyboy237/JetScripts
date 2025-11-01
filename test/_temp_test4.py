from typing import Literal
from jet.adapters.stanza.semgrex_visualization import adjust_dep_arrows, process_sentence_html
from jet.file.utils import save_file
import stanza
import spacy
from spacy import displacy
from spacy.tokens import Doc
from stanza.models.common.constant import is_right_to_left
from stanza.server.semgrex import Semgrex
from jet.transformers.formatters import format_json
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

text = """
##### Headhunted to Another World: From Salaryman to Big Four!

Isekai Fantasy Comedy
Release Date
: January 1, 2025
Japanese Title
: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi
Studio
: Geek Toys, CompTown
Based On
: Manga
Creator
: Benigashira
Streaming Service(s)
: Crunchyroll
Powered by Expand Collapse
Plenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.
Related
"""

def get_sentences_html(doc, language, style: Literal["dep", "ent", "span"] = "dep"):
    """
    Returns a list of the HTML strings of the dependency visualizations of a given stanza doc object.

    The 'language' arg is the two-letter language code for the document to be processed.

    First converts the stanza doc object to a spacy doc object and uses displacy to generate an HTML
    string for each sentence of the doc object.
    """
    html_strings = []

    nlp = spacy.blank("en")
    sentences_to_visualize = []
    for sentence in doc.sentences:
        words, lemmas, heads, deps, tags = [], [], [], [], []
        if is_right_to_left(language):  # order of words displayed is reversed, dependency arcs remain intact
            sent_len = len(sentence.words)
            for word in reversed(sentence.words):
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if word.head == 0:  # spaCy head indexes are formatted differently than that of Stanza
                    heads.append(sent_len - word.id)
                else:
                    heads.append(sent_len - word.head)
        else:  # left to right rendering
            for word in sentence.words:
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if word.head == 0:
                    heads.append(word.id - 1)
                else:
                    heads.append(word.head - 1)
        document_result = Doc(nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags)
        sentences_to_visualize.append(document_result)

    for line in sentences_to_visualize:  # render all sentences through displaCy
        html_strings.append(
            displacy.render(
                line,
                style=style,
                options={
                    "compact": True,
                    "word_spacing": 30,
                    "distance": 100,
                    "arrow_spacing": 20
                },
                jupyter=False
            )
        )
    return html_strings, [s.to_dict() for s in sentences_to_visualize]
    
def main():
    pipe = stanza.Pipeline("en", processors="tokenize, pos, lemma, depparse")
    doc = pipe(text)
    semgrex_queries = [
        "{pos:NN}=object <obl {}=action",
        "{cpos:NOUN}=thing <obj {cpos:VERB}=action",
    ]
    lang_code = "en"
    start_match = 0
    end_match = 10

    matches_count = 0  # Limits number of visualizations
    with Semgrex(classpath="$CLASSPATH") as sem:
        edited_html_strings = []
        semgrex_result_strings = []
        semgrex_results = sem.process(doc, *semgrex_queries)
        unedited_html_strings, nlp_info = get_sentences_html(doc, lang_code)
        for i in range(len(unedited_html_strings)):

            if matches_count >= end_match:  # we've collected enough matches, stop early
                break

            has_none = True
            for query in semgrex_results.result[i].result:
                for match in query.match:
                    if match:
                        has_none = False
                        semgrex_result_strings.append(match)

            if not has_none:
                if start_match <= matches_count < end_match:
                    edited_string = process_sentence_html(unedited_html_strings[i], semgrex_results.result[i])
                    edited_string = adjust_dep_arrows(edited_string)
                    edited_html_strings.append(edited_string)
                matches_count += 1

    logger.success(format_json(semgrex_result_strings))

    save_file(semgrex_result_strings, f"{OUTPUT_DIR}/semgrex_result_strings.json")
    save_file(semgrex_results, f"{OUTPUT_DIR}/semgrex_results.json")
    save_file(nlp_info, f"{OUTPUT_DIR}/nlp_info.json")

    for i, html in enumerate(unedited_html_strings):
        save_file(html, f"{OUTPUT_DIR}/unedited/semgrex{i + 1}.html")

    for i, html in enumerate(edited_html_strings):
        save_file(html, f"{OUTPUT_DIR}/edited/semgrex{i + 1}.html")

if __name__ == "__main__":
    main()
