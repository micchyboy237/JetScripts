import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, MarianMTModel, MarianTokenizer
from cache import InMemoryCache
from lightning.pytorch import LightningModule
from typing import Optional
from jet.logger import time_it

cache = InMemoryCache()


class TranslationModel(LightningModule):
    def __init__(
        self,
        model_name
    ):
        super().__init__()
        self.model = MarianMTModel.from_pretrained(model_name)


@time_it
def load_model_from_checkpoint(
    model_name,
    checkpoint_path,
):
    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model = TranslationModel(
        model_name=model_name,
    )
    model.load_state_dict(checkpoint['state_dict'])

    # Prepare tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    return model.model, tokenizer


def load_model(model_name: str, checkpoint_path: Optional[str] = None):
    """Load a model and its tokenizer from the given path using MarianMT.
    If the model is a file ending with ".ckpt" or ".pt", load the checkpoint.
    If the model has already been loaded, retrieve it from the cache.
    """
    model_tokenizer_pair = cache.models.get(
        checkpoint_path if checkpoint_path else model_name)
    if model_tokenizer_pair:
        # print(f"Cache hit! Retrieving model and tokenizer from cache.")
        model, tokenizer = model_tokenizer_pair
    else:
        print(
            f"Loading model and tokenizer from {checkpoint_path if checkpoint_path else model_name}")

        # Check if the checkpoint_path is a checkpoint file
        if checkpoint_path and (checkpoint_path.endswith('.ckpt') or checkpoint_path.endswith('.pt')):
            model, tokenizer = load_model_from_checkpoint(
                model_name, checkpoint_path)
        else:
            # Load the model as usual
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        cache.setModels(model_name, (model, tokenizer))

    return model, tokenizer


def load_translation_model(model_name: str, checkpoint_path: Optional[str] = None):
    """Load a model and its tokenizer from the given path using MarianMT.
    If the model is a file ending with ".ckpt" or ".pt", load the checkpoint.
    If the model has already been loaded, retrieve it from the cache.
    """
    model_tokenizer_pair = cache.models.get(
        checkpoint_path if checkpoint_path else model_name)
    if model_tokenizer_pair:
        # print(f"Cache hit! Retrieving model and tokenizer from cache.")
        model, tokenizer = model_tokenizer_pair
    else:
        print(
            f"Loading model and tokenizer from {checkpoint_path if checkpoint_path else model_name}")

        # Check if the checkpoint_path is a checkpoint file
        if checkpoint_path and (checkpoint_path.endswith('.ckpt') or checkpoint_path.endswith('.pt')):
            model, tokenizer = load_model_from_checkpoint(
                model_name, checkpoint_path)
        else:
            # Load the model as usual
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        cache.setModels(model_name, (model, tokenizer))

    return model, tokenizer


def load_gpt_model(model_path: str):
    """Load a GPT model and its tokenizer from the given path.
    If the model is a file ending with ".ckpt" or ".pt", load the checkpoint.
    If the model has already been loaded, retrieve it from the cache.
    """
    model_tokenizer_pair = cache.models.get(model_path)
    if model_tokenizer_pair:
        # print(f"Cache hit! Retrieving model and tokenizer from cache.")
        model, tokenizer = model_tokenizer_pair
    else:
        print(f"Loading model and tokenizer from {model_path}")

        # Check if the model_path is a checkpoint file
        if model_path.endswith('.ckpt') or model_path.endswith('.pt'):
            # Load the PyTorch checkpoint
            checkpoint = torch.load(
                model_path, map_location=torch.device("cpu"))

            # Assume that we need to create the model architecture here
            # Load with the given config
            model = GPT2LMHeadModel(checkpoint['config'])
            model.load_state_dict(checkpoint['state_dict'])
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint['tokenizer_name'])
        else:
            # Load the model as usual
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        model.eval()
        cache.setModels(model_path, (model, tokenizer))

    return model, tokenizer


@time_it
def save_model(model, tokenizer, model_path: str):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


@time_it
def save_model_to_checkpoint(model, checkpoint_dir, checkpoint_filename):
    """
    Save a model to a checkpoint file.
    """
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Save the state dictionary of the model
    model_state_dict = model.state_dict()

    # Create the checkpoint dictionary
    checkpoint = {
        'state_dict': model_state_dict,
        # You can include other configurations if necessary
    }

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


@time_it
def save_translation_model(model_name: str, checkpoint_dir: str, checkpoint_filename: str):
    """
    Save a model and its tokenizer to the given path using MarianMT.
    """
    print(f"Loading model from {model_name}")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Saving model to {checkpoint_dir}")
    if checkpoint_filename.endswith('.ckpt') or checkpoint_filename.endswith('.pt'):
        save_model_to_checkpoint(
            model, checkpoint_dir, checkpoint_filename)

    else:
        # If not saving as a checkpoint, define the full path for the model
        model_path = os.path.join(checkpoint_dir, checkpoint_filename)
        save_model(model, tokenizer, model_path)


@time_it
def load_tokenizer(tokenizer_name_or_path: str):
    tokenizer = cache.tokenizers.get(tokenizer_name_or_path)
    if tokenizer is None:
        print(f"Loading tokenizer from {tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        cache.setTokenizers(tokenizer_name_or_path, tokenizer)

    return tokenizer


@time_it
def count_tokens(text: str, tokenizer_name_or_path: str):
    """Counts the number of tokens in a text using the provided tokenizer."""
    tokenizer = load_tokenizer(tokenizer_name_or_path)

    decoded_text = tokenizer.encode(text)

    return len(decoded_text)
