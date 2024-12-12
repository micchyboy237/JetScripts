from datasets import load_dataset

from aim import Run
from aim.hf_dataset import HFDataset

# create dataset object
dataset = load_dataset('rotten_tomatoes')

# store dataset metadata
run = Run()
run['datasets_info'] = HFDataset(dataset)
