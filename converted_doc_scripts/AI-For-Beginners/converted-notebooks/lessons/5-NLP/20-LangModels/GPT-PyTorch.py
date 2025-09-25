from jet.logger import logger
from transformers import pipeline
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
## Experimenting with Ollama GPT

This notebook is part of [AI for Beginners Curriculum](http://aka.ms/ai-beginners).

In this notebook, we will explore how we can play with Ollama-GPT model using Hugging Face `transformers` library.

Without further ado, let's instantiate text generating pipeline and start generating!
"""
logger.info("## Experimenting with Ollama GPT")


model_name = 'ollama-gpt'

generator = pipeline('text-generation', model=model_name)

generator("Hello! I am a neural network, and I want to say that", max_length=100, num_return_sequences=5)

"""
## Prompt Engineering

In some of the problems, you can use ollama-gpt generation right away by designing correct prompts. Have a look at the examples below:
"""
logger.info("## Prompt Engineering")

generator("Synonyms of a word cat:", max_length=20, num_return_sequences=5)

generator("I love when you say this -> Positive\nI have myself -> Negative\nThis is awful for you to say this ->", max_length=40, num_return_sequences=5)

generator("Translate English to French: cat => chat, dog => chien, student => ", top_k=50, max_length=30, num_return_sequences=3)

generator("People who liked the movie The Matrix also liked ", max_length=40, num_return_sequences=5)

"""
## Text Sampling Strategies

So far we have been using simple **greedy** sampling strategy, when we selected next word based on the highest probability. Here is how it works:
"""
logger.info("## Text Sampling Strategies")

prompt = "It was early evening when I can back from work. I usually work late, but this time it was an exception. When I entered a room, I saw"
generator(prompt,max_length=100,num_return_sequences=5)

"""
**Beam Search** allows the generator to explore several directions (*beams*) of text generation, and select the ones with highers overall score. You can do beam search by providing `num_beams` parameter. You can also specify `no_repeat_ngram_size` to penalize the model for repeating n-grams of a given size:
"""

prompt = "It was early evening when I can back from work. I usually work late, but this time it was an exception. When I entered a room, I saw"
generator(prompt,max_length=100,num_return_sequences=5,num_beams=10,no_repeat_ngram_size=2)

"""
**Sampling** selects the next word non-deterministically, using the probability distribution returned by the model. You turn on sampling using `do_sample=True` parameter. You can also specify `temperature`, to make the model more or less deterministic.
"""

prompt = "It was early evening when I can back from work. I usually work late, but this time it was an exception. When I entered a room, I saw"
generator(prompt,max_length=100,do_sample=True,temperature=0.8)

"""
We can also provide to additional parameters to sampling:
* `top_k` specifies the number of word options to consider when using sampling. This minimizes the chance of getting weird (low-probability) words in our text.
* `top_p` is similar, but we chose the smallest subset of most probable words, whose total probability is larger than p.

Feel free to experiment with adding those parameters in.

## Fine-Tuning your models

You can also [fine-tune your model](https://learn.microsoft.com/en-us/azure/cognitive-services/ollama/how-to/fine-tuning?pivots=programming-language-studio?WT.mc_id=academic-77998-bethanycheum) on your own dataset. This will allow you to adjust the style of text, while keeping the major part of language model.
"""
logger.info("## Fine-Tuning your models")

logger.info("\n\n[DONE]", bright=True)