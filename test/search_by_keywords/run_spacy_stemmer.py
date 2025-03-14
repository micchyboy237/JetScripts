from jet.file.utils import load_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.words import get_spacy_words
from shared.data_types.job import JobData


if __name__ == "__main__":
    text = "React.js and JavaScript are used in web development."
    results = get_spacy_words(text)

    logger.newline()
    logger.debug(f"Results 1 ({len(results)})")
    logger.success(format_json(results))

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    selected_ids = ["82778617"]
    data = [d for d in data if d["id"] in selected_ids]

    sentences = [
        "\n".join([
            item["title"],
            item["details"],
            "\n".join([
                f"Tech: {tech}"
                for tech in sorted(
                    item["entities"]["technology_stack"],
                    key=str.lower
                )
            ]),
            "\n".join([
                f"Tag: {tech}"
                for tech in sorted(
                    item["tags"],
                    key=str.lower
                )
            ]),
        ])
        for item in data
    ]

    for idx, sentence in enumerate(sentences):
        results = get_spacy_words(sentence)

        logger.newline()
        logger.debug(f"Sentence {idx + 1} ({len(results)})")
        logger.success(format_json(results))
