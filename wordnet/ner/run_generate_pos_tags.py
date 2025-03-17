from jet.file.utils import load_file, save_file
from jet.utils.object import extract_values_by_paths
from shared.data_types.job import JobData
from tqdm import tqdm
from jet.wordnet.n_grams import get_common_texts


if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file)

    attributes = ["title", "details", "entities.technology_stack", "tags"]

    texts = []
    for item in data:
        data_parts = extract_values_by_paths(
            item, attributes, is_flattened=True)
        texts.append(
            "\n".join([
                data_parts["title"],
                data_parts["details"],
                "\n".join(data_parts["technology_stack"]),
                "\n".join(data_parts["tags"]),
            ])
        )

    # Get common texts
    tokens_to_visualize = get_common_texts(texts)

    save_file(tokens_to_visualize, "generated/jobs_most_common_texts.json")
