from gensim.similarities.annoy import AnnoyIndexer
from gensim.models import Word2Vec
from instruction_generator.utils.logger import logger


if __name__ == '__main__':
    model_path = 'generated/gensim_jet_word_model.pkl'
    topn = 5

    print(f"Loading model from {model_path}")
    model = Word2Vec.load(model_path)

    print("Indexing model...")
    indexer = AnnoyIndexer(model, 2)

    while True:
        base_word = input("Enter any word: ")
        # quit on 'exit' or ctrl+c
        if base_word == 'exit' or base_word == KeyboardInterrupt:
            logger.info("Exiting...")
            break

        try:
            results = model.wv.most_similar(
                base_word, topn=topn, indexer=indexer)
            print(f"Most similar words to '{base_word}':\n{results}")
        except KeyError:
            print(f"'{base_word}' not found in the model.")
