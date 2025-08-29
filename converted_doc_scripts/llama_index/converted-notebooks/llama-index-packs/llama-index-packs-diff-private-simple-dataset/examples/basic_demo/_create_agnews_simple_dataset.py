from jet.logger import CustomLogger
from llama_index.core.llama_dataset.base import CreatedBy, CreatedByType
from llama_index.core.llama_dataset.simple import (
    LabelledSimpleDataExample,
    LabelledSimpleDataset,
)
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Creating the AGNEWs `LabelledSimpleDataset`

In this notebook, we take the AGNEWs dataset ([original source](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)) and turn it into a `LabelledSimpleDataset` that we ultimately run the `DiffPrivateSimpleDatasetPack` on.
"""
logger.info("# Creating the AGNEWs `LabelledSimpleDataset`")


"""
### Load data

The dataset is also available from our public Dropbox.
"""
logger.info("### Load data")

# !mkdir -p "data/agnews/"
# !wget "https://www.dropbox.com/scl/fi/wzcuxuv2yo8gjp5srrslm/train.csv?rlkey=6kmofwjvsamlf9dj15m34mjw9&dl=1" -O "data/agnews/train.csv"

df = pd.read_csv(f"{os.path.dirname(__file__)}/data/agnews/train.csv")

class_to_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

df["Label"] = df["Class Index"].map(class_to_label)

df.head()

"""
### Create LabelledSimpleDataExample
"""
logger.info("### Create LabelledSimpleDataExample")

examples = []
for index, row in df.iterrows():
    example = LabelledSimpleDataExample(
        reference_label=row["Label"],
        text=f"{row['Title']} {row['Description']}",
        text_by=CreatedBy(type=CreatedByType.HUMAN),
    )
    examples.append(example)

simple_dataset = LabelledSimpleDataset(examples=examples)

simple_dataset.to_pandas()[:5]

simple_dataset.save_json("agnews.json")

logger.info("\n\n[DONE]", bright=True)
