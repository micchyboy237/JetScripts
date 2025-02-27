from jet.file.utils import save_file, load_file
from jet.logger import logger
from shared.data_types.job import JobData
data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
data: list[JobData] = load_file(data_file) or []

filtered_data = [
    item for item in data
    if "React Native" in item["entities"]["technology_stack"]
]


# Print sample output
logger.log("Filtered jobs with React Native:", len(
    filtered_data), colors=["GRAY", "SUCCESS"])
