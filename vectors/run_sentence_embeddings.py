from jet.llm.embeddings.sentence_embedding import Tokenizer, SentenceEmbedding
from jet.logger import logger

if __name__ == '__main__':
    util = SentenceEmbedding('all-MiniLM-L6-v2')
    text = "This is the first sample sentence."
    texts = [
        "This is the first sample sentence.",
        "Another sentence to encode and count tokens.",
        "Short text."
    ]

    tokenizer = util.get_tokenizer()
    print(f"Tokenizer type: {type(tokenizer).__name__}")

    emb_single = util.generate_embeddings(text)
    emb_list = util.generate_embeddings(texts)
    print(f"Single embedding length: {len(emb_single)}")
    print(f"Batch embedding shape: ({len(emb_list)}, {len(emb_list[0])})")

    count1 = util.get_token_counts(text)
    count2 = util.get_token_counts(texts)
    alt_count1 = util.get_token_counts_alt(text)
    alt_count2 = util.get_token_counts_alt(texts)
    logger.info("Token counts with batch_encode_plus:")
    logger.success(f"Single: {count1}")
    for t, c in zip(texts, count2):
        logger.success(f"Text: {t}\nToken count: {c}\n")
    logger.info("Token counts with direct tokenizer:")
    logger.success(f"Single: {alt_count1}")
    for t, c in zip(texts, alt_count2):
        logger.success(f"Text: {t}\nToken count: {c}\n")

    token_ids_1 = util.tokenize(text)
    token_ids_2 = util.tokenize(texts)
    print(f"Token IDs (single): {token_ids_1}")
    print(f"Token IDs (list): {token_ids_2}")

    readable_1 = util.tokenize_strings(text)
    readable_2 = util.tokenize_strings(texts)
    print(f"Tokens (single): {readable_1}")
    print(f"Tokens (list): {readable_2}")

    fn = util.get_tokenizer_fn()
    print(f"get_tokenizer_fn single: {fn(text)}")
    print(f"get_tokenizer_fn list: {fn(texts)}")

    logger.info("Using get_embedding_function with 'all-MiniLM-L6-v2'")
    embed_fn = util.get_embedding_function('all-MiniLM-L6-v2')
    emb_fn_single = embed_fn(text)
    emb_fn_list = embed_fn(texts)
    print(f"[get_embedding_function] Single length: {len(emb_fn_single)}")
    print(
        f"[get_embedding_function] List shape: ({len(emb_fn_list)}, {len(emb_fn_list[0])})")
