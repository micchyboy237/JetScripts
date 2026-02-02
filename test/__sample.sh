#!/bin/bash

# setup_smolagents_html_extractor.sh
# Shell script to create the smolagents_html_extractor project with all source files and README

set -e

PROJECT_DIR="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/smolagents/jet_examples/smolagents_html_extractor"

mkdir -p "$PROJECT_DIR"

cat > "$PROJECT_DIR/requirements.txt" <<EOF
smolagents>=0.2.0
beautifulsoup4>=4.12.0
requests>=2.31.0
pytest>=7.4.0
EOF

cat > "$PROJECT_DIR/tools.py" <<'EOF'
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import json
from pathlib import Path

def chunk_html(
    html_content: str,
    window_size: int = 4000,
    overlap: int = 800
) -> List[Dict[str, Any]]:
    """Split HTML into overlapping text chunks"""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        chunk_text = text[start:end]
        
        # Try to end at a space to avoid cutting words
        if end < len(text):
            last_space = chunk_text.rfind(' ')
            if last_space > 0:
                end = start + last_space
                chunk_text = text[start:end]
        
        chunks.append({
            "index": len(chunks),
            "start_char": start,
            "end_char": end,
            "text": chunk_text
        })
        
        start = end - overlap
        if start >= len(text):
            break
            
    return chunks


def extract_relevant_content(
    chunk_text: str,
    query: str,
    max_items_per_chunk: int = 5
) -> List[Dict[str, Any]]:
    """
    Dummy extraction function - in real use this would call an LLM
    Here we just do simple keyword matching for demonstration
    """
    query_words = set(query.lower().split())
    sentences = chunk_text.split('. ')
    
    results = []
    for i, sentence in enumerate(sentences):
        if len(results) >= max_items_per_chunk:
            break
        sentence_lower = sentence.lower()
        score = sum(1 for word in query_words if word in sentence_lower)
        if score >= 1:  # at least one matching word
            results.append({
                "chunk_index": None,  # filled later
                "sentence_index": i,
                "text": sentence.strip() + ".",
                "score": score,
                "relevance": "high" if score >= 3 else "medium"
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)


def format_final_results(results: List[Dict]) -> str:
    """Format all collected results into readable output"""
    if not results:
        return "No relevant content found."
        
    lines = ["Found relevant content:", ""]
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. [{item['relevance'].upper()}] Score: {item['score']}")
        lines.append(f"   {item['text']}")
        lines.append("")
    return "\n".join(lines)
EOF

cat > "$PROJECT_DIR/checkpoint.py" <<'EOF'
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_file = self.checkpoint_dir / "results.json"
        self.progress_file = self.checkpoint_dir / "progress.json"

    def save_progress(self, processed_chunks: int, total_chunks: int):
        data = {
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "last_updated": str(Path().absolute())
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_partial_results(self, results: List[Dict[str, Any]]):
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def load_progress(self) -> Optional[Dict]:
        if not self.progress_file.exists():
            return None
        with open(self.progress_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_results(self) -> List[Dict[str, Any]]:
        if not self.results_file.exists():
            return []
        with open(self.results_file, "r", encoding="utf-8") as f:
            return json.load(f)
EOF

cat > "$PROJECT_DIR/extractor.py" <<'EOF'
from smolagents import CodeAgent, Tool
from typing import List, Dict, Any
import requests
from .tools import chunk_html, extract_relevant_content, format_final_results
from .checkpoint import CheckpointManager

def run_html_extraction_pipeline(
    url_or_html: str,
    query: str,
    window_size: int = 4000,
    overlap: int = 800,
    resume: bool = True
) -> str:
    """
    Main function to extract relevant content from long HTML with progress tracking
    """
    checkpoint = CheckpointManager()

    # Load or fetch HTML
    if url_or_html.startswith(("http://", "https://")):
        print(f"Fetching URL: {url_or_html}")
        response = requests.get(url_or_html, timeout=15)
        response.raise_for_status()
        html_content = response.text
    else:
        html_content = url_or_html

    # Chunking
    print("Chunking HTML...")
    chunks = chunk_html(html_content, window_size, overlap)
    print(f"Created {len(chunks)} chunks")

    # Load previous state if resuming
    start_idx = 0
    all_results = []

    if resume:
        progress = checkpoint.load_progress()
        if progress:
            start_idx = progress["processed_chunks"]
            all_results = checkpoint.load_results()
            print(f"Resuming from chunk {start_idx}/{len(chunks)}")

    # Process chunks
    for i in range(start_idx, len(chunks)):
        chunk = chunks[i]
        print(f"Processing chunk {i+1}/{len(chunks)} (characters {chunk['start_char']}-{chunk['end_char']})")

        partial = extract_relevant_content(chunk["text"], query)
        for item in partial:
            item["chunk_index"] = i

        all_results.extend(partial)

        # Save progress after each chunk
        checkpoint.save_partial_results(all_results)
        checkpoint.save_progress(i + 1, len(chunks))

        print(f"→ Found {len(partial)} items in chunk {i+1}")

    # Final result
    final_output = format_final_results(all_results)
    print("\nExtraction complete!")
    return final_output


# --- For smolagents integration (optional) ---

chunk_tool = Tool(
    name="chunk_html",
    description="Split HTML into overlapping text chunks",
    func=chunk_html,
    input_schema={
        "html_content": {"type": "string"},
        "window_size": {"type": "integer", "default": 4000},
        "overlap": {"type": "integer", "default": 800}
    }
)

extract_tool = Tool(
    name="extract_relevant",
    description="Extract relevant sentences from a text chunk based on query",
    func=extract_relevant_content,
    input_schema={
        "chunk_text": {"type": "string"},
        "query": {"type": "string"},
        "max_items_per_chunk": {"type": "integer", "default": 5}
    }
)

# You could then create a CodeAgent with these tools if you want
# agent = CodeAgent(tools=[chunk_tool, extract_tool], model=...)
EOF

cat > "$PROJECT_DIR/test_extractor.py" <<'EOF'
import pytest
from extractor.tools import chunk_html, extract_relevant_content, format_final_results
from extractor.checkpoint import CheckpointManager
from pathlib import Path
import shutil

@pytest.fixture
def sample_html():
    return """
    <html><body>
    <h1>Test Document</h1>
    <p>This is a test paragraph about Python programming.</p>
    <p>Python is great for data science and web development.</p>
    <p>Artificial intelligence is transforming the world.</p>
    <p>Another sentence without any keywords.</p>
    </body></html>
    """.strip()


def test_chunk_html(sample_html):
    chunks = chunk_html(sample_html, window_size=100, overlap=30)
    assert len(chunks) >= 2
    assert all("text" in c for c in chunks)
    assert chunks[0]["start_char"] == 0
    # Use BeautifulSoup to get text length
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(sample_html, 'html.parser')
    plain_text = soup.get_text(separator=' ', strip=True)
    assert chunks[-1]["end_char"] <= len(plain_text)


def test_extract_relevant(sample_html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(sample_html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    results = extract_relevant_content(text, query="python programming")
    assert len(results) > 0
    assert any("Python" in r["text"] for r in results)
    assert all(r["score"] >= 1 for r in results)


def test_format_results():
    fake_results = [
        {"text": "Python is great.", "score": 3, "relevance": "high"},
        {"text": "Another sentence.", "score": 1, "relevance": "medium"}
    ]
    output = format_final_results(fake_results)
    assert "Found relevant content" in output
    assert "PYTHON IS GREAT" in output.upper()


def test_checkpoint(tmp_path):
    checkpoint = CheckpointManager(str(tmp_path))
    
    results = [{"text": "test", "score": 1}]
    checkpoint.save_partial_results(results)
    checkpoint.save_progress(5, 20)
    
    loaded_results = checkpoint.load_results()
    loaded_progress = checkpoint.load_progress()
    
    assert len(loaded_results) == 1
    assert loaded_progress["processed_chunks"] == 5
    assert loaded_progress["total_chunks"] == 20


def test_full_pipeline(sample_html):
    from extractor.extractor import run_html_extraction_pipeline
    result = run_html_extraction_pipeline(
        sample_html,
        query="python",
        window_size=150,
        overlap=40,
        resume=False
    )
    assert "Python" in result
EOF

cat > "$PROJECT_DIR/README.md" <<'EOF'
# smolagents_html_extractor

A modular, testable pipeline to process and extract relevant passages from long HTML documents using `smolagents`-style tools. Includes chunking, extraction, checkpointing, and unit tests.

## Project Structure

```
smolagents_html_extractor/
├── extractor.py          # main pipeline logic
├── tools.py              # custom tools (chunking, extraction, formatting)
├── checkpoint.py         # checkpoint management
├── test_extractor.py     # unit tests
└── requirements.txt
```

## How to Use

### Install dependencies

```sh
pip install -r requirements.txt
```

### Run all tests

```sh
pytest test_extractor.py -v
```

### Example: Extract passages from a URL

```sh
python -c "
from extractor.extractor import run_html_extraction_pipeline
print(run_html_extraction_pipeline(
    'https://example.com/long-page.html',
    'artificial intelligence',
    window_size=3500,
    overlap=700
))
"
```

### Customization

- Adjust `extract_relevant_content()` in `tools.py` for smarter extraction logic, or replace with an LLM call.
- Checkpoints and interim results are stored in the `./checkpoints/` directory.

### Test Coverage

Unit tests exercise chunking, extraction, checkpointing, and the end-to-end pipeline.

---

If you need to (re)generate the project, run:

```sh
bash setup_smolagents_html_extractor.sh
```

Enjoy!
EOF

echo "Project structure created in $PROJECT_DIR. See $PROJECT_DIR/README.md for details."