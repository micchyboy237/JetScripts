import os
import ctranslate2
import sentencepiece as spm

DEFAULT_CT2_MODEL_DIR = os.path.expanduser("~/.cache/hf_ctranslate2_models/ct2-opus-ja-en")

generator = ctranslate2.Generator(DEFAULT_CT2_MODEL_DIR)
sp = spm.SentencePieceProcessor("tokenizer.model")

prompt = "What is the meaning of life?"
prompt_tokens = sp.encode(prompt, out_type=str)

step_results = generator.generate_tokens(
    prompt_tokens,
    sampling_temperature=0.8,
    sampling_topk=20,
    max_length=1024,
)

output_ids = []

for step_result in step_results:
    is_new_word = step_result.token.startswith("▁")

    if is_new_word and output_ids:
        word = sp.decode(output_ids)
        print(word, end=" ", flush=True)
        output_ids = []

    output_ids.append(step_result.token_id)

if output_ids:
    word = sp.decode(output_ids)
    print(word)