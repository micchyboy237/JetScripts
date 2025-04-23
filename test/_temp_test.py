from transformers import pipeline
from jet.logger import logger
from jet.transformers.formatters import format_json


def sample_zero_shot_classification():
    classifier = pipeline("zero-shot-classification",
                          model="tasksource/ModernBERT-base-nli")

    text = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    classification = classifier(text, candidate_labels)

    logger.success(format_json(classification))


def sample_text_classification():
    classifier = pipeline(
        "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    text = "I love using this new app! It's amazing and saves so much time."
    result = classifier(text)

    logger.success(format_json(result))


def sample_named_entity_recognition():
    ner = pipeline("ner", grouped_entities=True)

    text = "Barack Obama was born in Hawaii and served as president of the United States."
    entities = ner(text)

    logger.success(format_json(entities))


def sample_summarization():
    summarizer = pipeline("summarization")

    article = (
        "The Amazon rainforest, also known as the Amazon jungle, is a moist broadleaf forest that covers most of the Amazon basin in South America. "
        "This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with Peru, Colombia, and smaller amounts in Venezuela, "
        "Ecuador, Bolivia, Guyana, Suriname, and French Guiana."
    )
    summary = summarizer(article, max_length=45,
                         min_length=15, do_sample=False)

    logger.success(format_json(summary))


def sample_translation():
    translator = pipeline("translation_en_to_fr",
                          model="Helsinki-NLP/opus-mt-en-fr")

    text = "The weather is beautiful today and perfect for a picnic in the park."
    translated = translator(text)

    logger.success(format_json(translated))


def sample_text_generation():
    generator = pipeline("text-generation", model="gpt2")

    prompt = "In a future dominated by artificial intelligence,"
    generated = generator(prompt, max_length=50, num_return_sequences=1)

    logger.success(format_json(generated))


def sample_question_answering():
    qa_pipeline = pipeline("question-answering")

    context = (
        "The Eiffel Tower is one of the most iconic landmarks in the world. "
        "It was constructed in 1889 and stands in Paris, France."
    )
    question = "When was the Eiffel Tower built?"

    answer = qa_pipeline(question=question, context=context)

    logger.success(format_json(answer))


def sample_fill_mask():
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    sentence = "The capital of France is [MASK]."
    predictions = fill_mask(sentence)

    logger.success(format_json(predictions))


if __name__ == "__main__":
    sample_zero_shot_classification()
    sample_text_classification()
    sample_named_entity_recognition()
    sample_summarization()
    sample_translation()
    sample_text_generation()
    sample_question_answering()
    sample_fill_mask()
