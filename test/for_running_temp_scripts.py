from jet.file.utils import save_file, load_file
from jet.logger import logger
from shared.data_types.job import JobData
from jet.scrapers.utils import clean_text
from tqdm import tqdm

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
data: list[JobData] = load_file(data_file) or []

for item in tqdm(data):
    item["title"] = clean_text(item["title"])
    item["details"] = clean_text(item["details"])
    item["tags"] = [clean_text(tag) for tag in item["tags"]]
    item["entities"] = {
        key: [clean_text(v) for v in value]
        for key, value in item["entities"].items()
    }

    save_file(data, data_file)
