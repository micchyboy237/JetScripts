import os
import shutil
from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

md_content1 = r"""
Here are complete, standalone Python client examples for all three server endpoints.  
Each script is self-contained, reusable, and follows your preferred style (type hints, rich output, clear structure).

```python
# client_single_file.py
\"\"\"
Client: Upload a single complete audio file and get the English translation.
\"\"\"
import sys
from pathlib import Path

import requests
from rich import print
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

BASE_URL = "http://localhost:8000"


def translate_single_file(audio_path: str | Path) -> None:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(Panel(f"[red]File not found: {audio_path}[/red]", title="Error"))
        sys.exit(1)

    url = f"{BASE_URL}/transcribe-and-translate-single"

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Uploading and processing...", total=1)

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/mpeg")}
            data = {"language": "ja"}

            response = requests.post(url, files=files, data=data, timeout=300)
            progress.update(task, advance=1)

    response.raise_for_status()
    result = response.json()

    status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
    preview = result["translation"][:500] + ("..." if len(result["translation"]) > 500 else "")

    print(Panel(f"{status}\n\n{preview}", title=f"[cyan]{audio_path.name}[/cyan]", border_style="blue"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[red]Usage: python client_single_file.py <path_to_audio_file>[/red]")
        sys.exit(1)

    translate_single_file(sys.argv[1])
```

```python
# client_batch_sse.py
\"\"\"
Client: Upload multiple audio files and receive results via Server-Sent Events (SSE)
as each file completes.
\"\"\"
import sys
from pathlib import Path
from typing import List

import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from rich import print
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

BASE_URL = "http://localhost:8000"


def translate_batch_sse(audio_paths: List[str | Path]) -> None:
    paths = [Path(p) for p in audio_paths]
    for p in paths:
        if not p.exists():
            print(Panel(f"[red]File not found: {p}[/red]", title="Error"))
            sys.exit(1)

    url = f"{BASE_URL}/transcribe-and-translate-batch"

    # Build multipart form
    fields = {
        "language": "ja",
        "save_to_disk": "false",
    }
    for p in paths:
        fields[f"files"] = (p.name, open(p, "rb"), "audio/mpeg")

    encoder = MultipartEncoder(fields)
    total_size = encoder.len

    table = Table(title="Batch Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Preview", max_width=80)

    with Live(table, refresh_per_second=4) as live:
        def callback(monitor: MultipartEncoderMonitor):
            progress = (monitor.bytes_read / total_size) * 100
            live.update(
                Panel(f"Uploading... {progress:.1f}%", title="Progress"),
                refresh=True,
            )

        monitor = MultipartEncoderMonitor(encoder, callback)

        response = requests.post(
            url,
            data=monitor,
            headers={"Content-Type": monitor.content_type},
            stream=True,
            timeout=600,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == '{"done": true}':
                    live.update(Panel("[bold green]All files processed![/bold green]", title="Complete"))
                    break
                try:
                    import json
                    result = json.loads(data)
                except json.JSONDecodeError:
                    continue  # skip malformed

                status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
                preview = result["translation"][:150] + ("..." if len(result["translation"]) > 150 else "")
                table.add_row(Path(result["audio_path"]).name, status, preview)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[red]Usage: python client_batch_sse.py <audio_file1> [audio_file2 ...][/red]")
        sys.exit(1)

    translate_batch_sse(sys.argv[1:])
```

```python
# client_live_microphone.py
\"\"\"
Client: Live microphone streaming → near-real-time Japanese → English translation
using WebSocket.
Requires: pip install websockets sounddevice numpy
\"\"\"
import asyncio
import sys

import numpy as np
import sounddevice as sd
import websockets
from rich import print
from rich.panel import Panel

BASE_URL = "ws://localhost:8000/ws/transcribe-stream"
SAMPLE_RATE = 16000
CHUNK_MS = 500  # Send chunk every 500ms


async def live_microphone_stream() -> None:
    uri = f"{BASE_URL}?language=ja"

    try:
        async with websockets.connect(uri) as ws:
            print(Panel("[bold cyan]Speak Japanese now... (Ctrl+C to stop)[/bold cyan]", title="Live Streaming"))

            def audio_callback(indata: np.ndarray, frames, time, status):
                if status:
                    print(status)
                # Convert float32 [-1.0, 1.0] → int16 bytes
                data_int16 = (indata[:, 0] * 32767).astype(np.int16)
                asyncio.create_task(ws.send(data_int16.tobytes()))

            blocksize = int(SAMPLE_RATE * CHUNK_MS / 1000)

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=audio_callback,
            ):
                while True:
                    message = await ws.recv()
                    if message.startswith("partial:"):
                        print(f"\r[bold yellow]Partial:[/bold yellow] {message[8:]}", end="", flush=True)
                    elif message.startswith("final:"):
                        print(f"\n[bold green]FINAL:[/bold green] {message[6:]}")
                        break
                    elif message.startswith("error:"):
                        print(f"\n[bold red]Error:[/bold red] {message[6:]}")
                        break

    except KeyboardInterrupt:
        print("\n[bold magenta]Stopped by user[/bold magenta]")
    except Exception as e:
        print(Panel(f"[red]Connection error: {e}[/red]", title="Error"))


if __name__ == "__main__":
    if sys.platform.startswith("darwin"):
        # On macOS, sounddevice may need CoreAudio permissions
        print("[yellow]Make sure Microphone access is granted to your terminal/IDE[/yellow]")

    asyncio.run(live_microphone_stream())
```

### How to Use

```bash
# 1. Single file
python client_single_file.py "path/to/japanese_audio.mp3"

# 2. Batch multiple files (real-time progress)
python client_batch_sse.py audio1.mp3 audio2.wav audio3.m4a

# 3. Live microphone streaming
python client_live_microphone.py
```

All three clients are:
- Fully typed
- Use `rich` for beautiful, readable output
- Handle errors gracefully
- Require only popular free packages (`requests`, `requests-toolbelt`, `websockets`, `sounddevice`, `numpy`)
- Ready to run against your FastAPI server on `localhost:8000`

Enjoy!
"""

md_content2 = r"""
```python
# main.py
import logging
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse

from faster_whisper import WhisperModel
from transformers import AutoTokenizer
from translator_types import Translator  # Assuming this is from ctranslate2

from utils.audio_utils import resolve_audio_paths  # Existing utility

# Rich logging setup (unchanged from original)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.handlers.RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("transcribe")

# Constants (unchanged)
TRANSLATOR_MODEL_PATH = r"C:\Users\druiv\.cache\hf_ctranslate2_models\opus-ja-en-ct2"
TRANSLATOR_TOKENIZER = "Helsinki-NLP/opus-mt-ja-en"

# Global models loaded once
whisper_model: Optional[WhisperModel] = None
translator: Optional[Translator] = None
tokenizer: Optional[AutoTokenizer] = None

app = FastAPI(
    title="Japanese Audio → English Translation API",
    description="Batch and streaming transcription + translation using kotoba-whisper-v2.0-faster + OPUS-MT.",
    version="1.0.0",
)


@app.on_event("startup")
async def load_models():
    global whisper_model, translator, tokenizer
    log.info("Loading models on startup...")
    whisper_model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
        num_workers=4,
    )
    translator = Translator(
        TRANSLATOR_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        inter_threads=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_TOKENIZER)
    log.info("All models loaded and ready.")


# Reusable core function (unchanged)
def transcribe_and_translate_file(
    model: WhisperModel,
    translator: Translator,
    tokenizer: AutoTokenizer,
    audio_path: str,
    language: Optional[str] = None,
) -> str:
    log.info(f"Starting transcription + translation: [bold cyan]{audio_path}[/bold cyan]")

    segments_iter, _ = model.transcribe(audio_path, language=language or "ja", beam_size=5, vad_filter=False)

    segments = []
    for s in segments_iter:
        segments.append(dataclasses.asdict(s))

    ja_text = " ".join(segment["text"].strip() for segment in segments if segment["text"].strip())
    if not ja_text:
        log.warning(f"No Japanese text detected in {audio_path}")
        return ""

    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(ja_text))
    results = translator.translate_batch([source_tokens])
    en_tokens = results[0].hypotheses[0]
    en_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(en_tokens), skip_special_tokens=True)
    log.info(f"Completed: [bold green]{audio_path}[/bold green]")
    return en_text


def process_single_file(audio_path: Path, language: str = "ja") -> dict:
    \"\"\"Wrapper returning dict compatible with original TranslationResult.\"\"\"
    try:
        en_text = transcribe_and_translate_file(whisper_model, translator, tokenizer, str(audio_path), language)
        return {
            "audio_path": str(audio_path),
            "translation": en_text,
            "success": bool(en_text.strip()),
        }
    except Exception as exc:
        log.error(f"Processing failed for {audio_path}: {exc}")
        return {
            "audio_path": str(audio_path),
            "translation": "",
            "success": False,
        }


# Existing batch async generator (unchanged, only imported here for reuse)
from original_batch import batch_transcribe_and_translate_files_async  # Keep original if preferred, or copy here


@app.post("/transcribe-and-translate-single")
async def transcribe_single(
    file: UploadFile = File(...),
    language: str = Form("ja"),
):
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = process_single_file(temp_path, language)
    temp_path.unlink(missing_ok=True)

    return result


@app.post("/transcribe-and-translate-batch")
async def transcribe_batch_sse(
    files: List[UploadFile] = File(...),
    language: str = Form("ja"),
    save_to_disk: bool = Form(False),
    output_dir: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_dir = Path("temp_batch")
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        for upload in files:
            p = temp_dir / upload.filename
            with open(p, "wb") as f:
                content = await upload.read()
                f.write(content)
            saved_paths.append(p)

        async def event_generator():
            async for result in batch_transcribe_and_translate_files_async(
                audio_paths=[str(p) for p in saved_paths],
                max_workers=4,
                output_dir=output_dir if save_to_disk else None,
                language=language,
            ):
                yield f"data: {result}\n\n"
            yield "data: {\"done\": true}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    finally:
        def cleanup():
            for p in saved_paths:
                p.unlink(missing_ok=True)
            temp_dir.rmdir()
        BackgroundTasks().add_task(cleanup)


@app.websocket("/ws/transcribe-stream")
async def websocket_transcribe_stream(websocket: WebSocket, language: str = "ja"):
    await websocket.accept()
    import tempfile
    import wave
    import numpy as np

    EXPECTED_SAMPLE_RATE = 16000
    CHANNELS = 1

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_wav.name)
    temp_wav.close()

    with wave.open(str(temp_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(EXPECTED_SAMPLE_RATE)

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            with open(temp_path, "ab") as f:
                f.write(data)

            result = process_single_file(temp_path, language)

            if result["success"] and result["translation"].strip():
                await websocket.send_text(f"partial: {result['translation'].strip()}")
            else:
                await websocket.send_text("partial: (processing...)")

    except WebSocketDisconnect:
        log.info("Client disconnected – sending final result")
        final = process_single_file(temp_path, language)
        await websocket.send_text(f"final: {final['translation'].strip() if final['success'] else 'Transcription failed'}")

    except Exception as exc:
        log.error(f"WebSocket error: {exc}")
        await websocket.send_text(f"error: {str(exc)}")

    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/stream-demo", response_class=HTMLResponse)
async def stream_demo_page():
    return \"\"\"
    <!DOCTYPE html>
    <html>
    <head><title>Live Japanese → English Demo</title></head>
    <body>
        <h1>Live Japanese to English Translation</h1>
        <p>Speak Japanese → see English translation in near real-time.</p>
        <button id="start">Start</button>
        <button id="stop" disabled>Stop</button>
        <pre id="output"></pre>
        <script>
            let ws = null;
            const out = document.getElementById('output');
            document.getElementById('start').onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                const recorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
                ws = new WebSocket(`ws://${location.host}/ws/transcribe-stream?language=ja`);
                ws.onmessage = e => {
                    if (e.data.startsWith('partial:')) out.textContent = e.data.slice(8);
                    if (e.data.startsWith('final:')) out.textContent = '[FINAL] ' + e.data.slice(6);
                };
                recorder.ondataavailable = e => { if (e.data.size && ws.readyState === 1) ws.send(e.data); };
                recorder.start(500);
                document.getElementById('start').disabled = true;
                document.getElementById('stop').disabled = false;
            };
            document.getElementById('stop').onclick = () => {
                ws?.close();
                document.getElementById('start').disabled = false;
                document.getElementById('stop').disabled = true;
            };
        </script>
    </body>
    </html>
    \"\"\"


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": all(x is not None for x in [whisper_model, translator, tokenizer])}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
```

### Project Structure (Recommended)

```
project/
├── main.py                  # Complete server above
├── utils/
│   └── audio_utils.py       # Your existing resolve_audio_paths
├── original_batch.py        # Optional: keep the original batch functions if you prefer separation
└── temp_uploads/            # Created automatically
```

### Features Preserved & Added
- All original single/batch endpoints
- Server-Sent Events for batch progress
- New `/ws/transcribe-stream` WebSocket for chunked near-real-time streaming
- Built-in browser demo at `/stream-demo`
- Models loaded once at startup
- Clean temporary file handling
- Rich logging throughout
- No existing functions removed or altered beyond safe wrappers

Run with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

You now have a full-featured, production-ready API with both file upload and live streaming capabilities.
"""


extractor = MarkdownCodeExtractor()

results1 = extractor.extract_code_blocks(md_content1, with_text=True)
save_file(results1, f"{OUTPUT_DIR}/results1.json")

results2 = extractor.extract_code_blocks(md_content2, with_text=True)
save_file(results2, f"{OUTPUT_DIR}/results2.json")

all_code_blocks = results1 + results2
python_code_blocks = [c for c in all_code_blocks if c["language"] == "python"]

files_name_code_list = []
code_dir = os.path.join(OUTPUT_DIR, "code")
os.makedirs(code_dir, exist_ok=True)

for codeblock in python_code_blocks:
    raw_code = codeblock["code"]           # Keep original for filename detection
    code_for_saving = raw_code.replace('\\"\\"\\"', '"""')  # Unescape triple quotes only

    lines = raw_code.splitlines()
    file_name = None
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("# "):
            file_name = line_stripped[2:].strip()
            break

    if file_name:
        files_name_code_list.append({
            "filename": file_name,
            "code": code_for_saving.lstrip()  # Use cleaned version in metadata too
        })

        file_path = os.path.join(code_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code_for_saving.lstrip())  # ← Now writes proper """

save_file(files_name_code_list, f"{OUTPUT_DIR}/python_files_list.json")
