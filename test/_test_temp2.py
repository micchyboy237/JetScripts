model_path = "/Users/jethroestrada/.cache/llama.cpp/LFM2-350M-ENJP-MT.Q4_K_M.gguf"

from llama_cpp import Llama
from rich.live import Live
from rich.text import Text


llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)

messages = [
    {"role": "system", "content": "Translate the following Japanese text to natural, fluent English."},
    {"role": "user", "content": "本商品は30日経過後の返品・交換はお受けできませんのでご了承ください。"}
]

stream = llm.create_chat_completion(
    messages=messages,
    temperature=0.6,
    min_p=0.08,
    top_p=0.94,
    repeat_penalty=1.05,
    max_tokens=768,
    stream=True
)

with Live(auto_refresh=False) as live:
    full_text = ""
    for chunk in stream:
        if "choices" not in chunk or not chunk["choices"]:
            continue
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            full_text += content
            live.update(Text(full_text))
            live.refresh()