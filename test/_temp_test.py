from typing import Final

SYSTEM_PROMPT: Final[str] = """You are a dirty-talking, porn-savvy translator, specializing in Japanese to English translations for adult films. Your mission: pornify the dialogue—make it sound like it’s straight out of a raunchy American adult movie, with all the filth, teasing, and over-the-top eroticism that implies.

Key Rules:
- Preserve the original intent, but amplify the sleaze: If the original is polite or subtle, make it flirtatious, teasing, or outright dirty. If it’s already explicit, go full-throttle with vulgar, graphic, and playful porn lingo.
- Match the escalation: Japanese adult videos often start with polite innuendo and build to raw, explicit sex talk. Your translation should follow this arc—teasing → flirtatious → filthy.
- Personality matters: A shy character should sound nervously naughty; a dominant should sound commanding and crude; a tease should be dripping with sarcasm and double entendres.
- Sex scenes = no holds barred: Use the raunchiest, most graphic, and most vulgar language possible. Think American porn clichés: “fuck me harder,” “take this big cock,” “you’re such a dirty slut,” etc.
- Fix AI errors: The subtitles are AI-generated and may have mistakes. Use context to correct them, but never invent new dialogue—just make what’s there sound as pornographic as possible.

Formatting:
- One-to-one line translation: Each original line must have exactly one translated line.
- Keep it concise: Subtitles must be readable on screen, so keep it short and punchy.
- Punctuation and grammar: Adapt to the target language, but prioritize pornographic impact over strict grammar.

Output Format:
#LINE_NUMBER
Original> [original text]
Translation> [pornified translation]

At the end, include:
<summary>
A one- or two-line synopsis of the current batch, emphasizing the erotic content.
</summary>
<scene>
A short, dirty summary of the current scene, including any previous batches.
</scene>"""

RETRY_PROMPT_SUFFIX: Final[str] = """
Your last translation wasn’t filthy enough!
Please translate the subtitles again, ensuring:
- Every line is translated separately—no merging!
- Every line is pornified—no vanilla dialogue allowed!
- Timing is preserved—keep the line count exact.
"""

from typing import List, TypedDict, Literal, Optional
from llama_cpp import Llama
from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("pornifier")

# TypedDicts (unchanged)
class SubtitleLine(TypedDict):
    number: int
    text: str

class TranslationResult(TypedDict):
    line_number: int
    original: str
    translation: str

class BatchResult(TypedDict):
    translations: List[TranslationResult]
    summary: str
    scene: str

def build_user_prompt(lines: List[SubtitleLine], is_retry: bool = False) -> str:
    """Builds the user message with numbered lines."""
    header = "Please translate the following subtitles to English.\n\n"
    if is_retry:
        header += RETRY_PROMPT_SUFFIX + "\n\n"
    
    content = header
    for line in lines:
        content += f"#{line['number']}\nOriginal> {line['text']}\n\n"
    return content.strip()

def parse_response(response_text: str) -> BatchResult:
    """Parses the expected structured output from the LLM."""
    lines = response_text.strip().splitlines()
    translations: List[TranslationResult] = []
    summary = ""
    scene = ""
    current_section: Optional[Literal["summary", "scene"]] = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") and line[1:].strip().isdigit():
            num_str = line[1:].strip()
            if i + 2 >= len(lines):
                break
            orig_line = lines[i + 1].strip()
            trans_line = lines[i + 2].strip()
            if orig_line.startswith("Original>") and trans_line.startswith("Translation>"):
                try:
                    num = int(num_str)
                    translations.append({
                        "line_number": num,
                        "original": orig_line.split(">", 1)[1].strip(),
                        "translation": trans_line.split(">", 1)[1].strip()
                    })
                except ValueError:
                    logger.warning(f"Skipping malformed line number: {num_str}")
            i += 3
            continue
        elif line == "<summary>":
            current_section = "summary"
            i += 1
            continue
        elif line == "</summary>":
            current_section = None
            i += 1
            continue
        elif line == "<scene>":
            current_section = "scene"
            i += 1
            continue
        elif line == "</scene>":
            current_section = None
            i += 1
            continue

        if current_section == "summary":
            summary += line + " "
        elif current_section == "scene":
            scene += line + " "
        i += 1

    return {
        "translations": translations,
        "summary": summary.strip(),
        "scene": scene.strip()
    }

def pornify_batch(
    lines: List[SubtitleLine],
    model_path: str,                          # Required: path to your .gguf file
    n_gpu_layers: int = -1,                   # -1 = all layers to GPU if possible
    n_ctx: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    is_retry: bool = False,
    max_retries: int = 2,
) -> BatchResult:
    """Main function: pornify one batch of subtitles using llama-cpp-python directly."""
    logger.info(f"Loading model from {model_path} (this may take a while first time)...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,                        # Set True for detailed llama.cpp logs
    )
    logger.info("Model loaded.")

    user_prompt = build_user_prompt(lines, is_retry)

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Generating (attempt {attempt + 1}/{max_retries + 1})...")
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                # stop=["</scene>"],  # optional: can help cut off early if needed
            )
            raw_output = response["choices"][0]["message"]["content"]
            if not raw_output:
                raise ValueError("Empty response from model")

            result = parse_response(raw_output)

            if len(result["translations"]) == len(lines):
                console.print(f"[green]Success:[/] Parsed {len(result['translations'])} lines")
                return result
            else:
                logger.warning(
                    f"Mismatch: expected {len(lines)} lines, got {len(result['translations'])}"
                )
                if attempt < max_retries:
                    console.print("[yellow]Retrying with filth boost...[/]")
                    is_retry = True

        except Exception as e:
            logger.error(f"Error during generation (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise

    raise ValueError("Failed to get valid structured output after retries")


from rich import print as rprint

def main():
    # Replace with your actual GGUF path
    YOUR_MODEL_PATH = r"C:\path\to\your\qwen2.5-7b-instruct-uncensored.Q5_K_M.gguf"

    # Example batch from a JAV scene
    sample_batch: list[SubtitleLine] = [
        {"number": 150, "text": "あの…恥ずかしいです…"},
        {"number": 151, "text": "もっと触ってください…"},
        {"number": 152, "text": "もう…ダメっ！"},
    ]

    try:
        result = pornify_batch(
            sample_batch,
            model_path=YOUR_MODEL_PATH,
            n_gpu_layers=-1,          # offload all possible to GPU
            n_ctx=8192,               # adjust to your model's max context
            temperature=0.75,
            max_tokens=2048,
        )
        rprint("[bold cyan]Translations:[/]")
        for t in result["translations"]:
            print(f"#{t['line_number']}")
            print(f"Original> {t['original']}")
            print(f"Translation> {t['translation']}\n")
        
        rprint("[bold magenta]Summary:[/]")
        print(result["summary"])
        rprint("[bold magenta]Scene:[/]")
        print(result["scene"])

    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()