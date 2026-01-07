from transformers import pipeline

model_name = "webbigdata/ALMA-7B-Ja"

console.print("Loading ALMA-7B-Ja")
translator = pipeline("translation", model=model_name, device_map="auto")

def translate_ja_en_alma(text: str) -> str:
    with tqdm(total=1, desc="Translating"):
        result = translator(text)
    return result[0]["translation_text"]

# Example
translation = translate_ja_en_alma(ja_text)
console.print("[bold green]Translation:[/bold green]", translation)