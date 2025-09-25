from datasets import load_dataset
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from jet.logger import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer
from transformers import AutoTokenizer
from transformers import TrainingArguments
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
id: huggingface
title: Hugging Face
sidebar_label: Hugging Face
---

## Quick Summary

Hugging Face provides developers with a comprehensive suite of pre-trained NLP models through its `transformers` library. To recap, here is how you can use Mistral's `mistralai/Mistral-7B-v0.1` model through Hugging Face's `transformers` library:
"""
logger.info("## Quick Summary")

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
logger.debug(tokenizer.batch_decode(generated_ids)[0])

"""
## Evals During Fine-Tuning

`deepeval` integrates with Hugging Face's `transformers.Trainer` module through the `DeepEvalHuggingFaceCallback`, enabling real-time evaluation of LLM outputs during model fine-tuning for each epoch.

:::info
In this section, we'll walkthrough an example of fine-tuning Mistral's 7B model.
:::

### Prepare Dataset for Fine-tuning
"""
logger.info("## Evals During Fine-Tuning")


training_dataset = load_dataset("text", data_files={"train": "train.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenized_dataset = training_dataset.map(tokenize_function, batched=True)

"""
### Setup Training Arguments
"""
logger.info("### Setup Training Arguments")

...

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

"""
### Initialize LLM and Trainer for Fine-Tuning
"""
logger.info("### Initialize LLM and Trainer for Fine-Tuning")

...

llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")


trainer = Trainer(
    model=llm,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

"""
### Define Evaluation Criteria

Use `deepeval` to define an `EvaluationDataset` and the metrics you want to evaluate your LLM on:
"""
logger.info("### Define Evaluation Criteria")


first_golden = Golden(input="...")
second_golden = Golden(input="...")

dataset = EvaluationDataset(goldens=[first_golden, second_golden])
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is coherent with the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

"""
:::info
We initialize an `EvaluationDataset` with [goldens instead of test cases](/docs/evaluation-datasets#with-goldens) since we're running inference at evaluation time.
:::

### Fine-tune and Evaluate

Then, create a `DeepEvalHuggingFaceCallback`:
"""
logger.info("### Fine-tune and Evaluate")

...

deepeval_hugging_face_callback = DeepEvalHuggingFaceCallback(
    evaluation_dataset=dataset,
    metrics=[coherence_metric],
    trainer=trainer
)

"""
The `DeepEvalHuggingFaceCallback` accepts the following arguments:

- `metrics`: the `deepeval` evaluation metrics you wish to leverage.
- `evaluation_dataset`: a `deepeval` `EvaluationDataset`.
- `aggregation_method`: a string of either 'avg', 'min', or 'max' to determine how metric scores are aggregated.
- `trainer`: a `transformers.trainer` instance.
- `tokenizer_args`: Arguments for the tokenizer.

Lastly, add `deepeval_hugging_face_callback` to your `transformers.Trainer`, and begin fine-tuning:
"""
logger.info("The `DeepEvalHuggingFaceCallback` accepts the following arguments:")

...
trainer.add_callback(deepeval_hugging_face_callback)

trainer.train()

"""
With this setup, evaluations will be ran on the entirety of your `EvaluationDataset` according to the metrics you defined at the end of each `epoch`.
"""
logger.info("With this setup, evaluations will be ran on the entirety of your `EvaluationDataset` according to the metrics you defined at the end of each `epoch`.")

logger.info("\n\n[DONE]", bright=True)