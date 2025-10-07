import os
import shutil
import logging
import difflib
import json
from typing import TypedDict, List, Literal
from jet.file.utils import save_file
from jet.wordnet.text_chunker import chunk_texts_with_data, chunk_texts_fast

# Optional: For Git-like unified diff summary (install with: pip install gitpython)
try:
    import git
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False
    git = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            os.path.dirname(__file__), "generated", "text_chunker.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class ChunkDict(TypedDict):
    content: str
    # Add other expected keys as needed; filtered dynamically

class LineRange(TypedDict):
    start: int
    end: int
    text: str

class DiffChange(TypedDict):
    status: Literal["add", "change", "remove"]
    from_range: LineRange
    to_range: LineRange

class DiffSummary(TypedDict):
    chunk_index: int
    regular_content: str
    fast_content: str
    unified_diff: List[DiffChange]

class SummaryData(TypedDict):
    timestamp: str
    input_length: int
    chunk_count: int
    methods_match: bool
    differences: List[DiffSummary]

def dict_to_string(d: dict) -> str:
    """Convert a dictionary to a formatted string for diff comparison."""
    return json.dumps(d, indent=2, sort_keys=True)

def html_escape(text: str) -> str:
    """Escape HTML special characters for safe <pre> display."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

def colorize_diff_line_with_substrings(line: str, is_addition: bool, is_deletion: bool, prev_line: str = "") -> str:
    """
    Colorize a diff line, highlighting only differing substrings using difflib.
    For additions/deletions, compare with prev_line to highlight changed parts.
    """
    if is_addition and not line.startswith('+++'):
        if prev_line.startswith('-'):
            # Compare with previous deletion line to highlight differences
            matcher = difflib.SequenceMatcher(None, prev_line[1:], line[1:])
            result = []
            for tag, _, _, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    result.append(html_escape(line[j1:j2]))
                elif tag in ('replace', 'insert'):
                    result.append(f'<span style="background-color: #99ff99;">{html_escape(line[j1:j2])}</span>')
            return f'+{''.join(result)}'
        return f'+<span style="background-color: #99ff99;">{html_escape(line[1:])}</span>'  # Green for additions
    elif is_deletion and not line.startswith('---'):
        if prev_line.startswith('+'):
            # Compare with next addition line (handled in next iteration)
            return f'-{html_escape(line[1:])}'
        return f'-<span style="background-color: #ff9999;">{html_escape(line[1:])}</span>'  # Red for deletions
    elif line.startswith('@@'):
        return f'<span style="color: #0000ff;">{html_escape(line)}</span>'  # Blue for hunk headers
    elif line.startswith('diff ') or line.startswith('--- ') or line.startswith('+++ '):
        return f'<span style="color: #808080;">{html_escape(line)}</span>'  # Gray for metadata
    return html_escape(line)  # No color for context lines

def generate_unified_diff_summary(
    from_lines: List[str], to_lines: List[str], path: str = "chunk"
) -> tuple[List[DiffChange], str]:
    """
    Generate a unified diff summary as both a list of typed dictionaries and a colored string
    with substring-level highlighting for differences.
    Falls back to difflib's unified_diff if GitPython unavailable or Git CLI fails.
    """
    changes: List[DiffChange] = []
    
    # Generate string-based diff for HTML output
    if GITPYTHON_AVAILABLE:
        from_content = "\n".join(from_lines) + "\n"
        to_content = "\n".join(to_lines) + "\n"
        input_content = f"{from_content}\n---\n{to_content}"
        import subprocess
        try:
            logger.debug("Running git diff --no-index with input length: %d", len(input_content))
            result = subprocess.run(
                ["git", "diff", "--no-index", "--unified=3", "-"],
                input=input_content,
                capture_output=True,
                text=True,
                check=True
            )
            diff_text = result.stdout if result.stdout.strip() else ""
            logger.debug("Git diff output length: %d", len(diff_text))
        except subprocess.CalledProcessError as e:
            logger.error("Git diff failed: %s", e.stderr)
            diff_text = None
        except FileNotFoundError:
            logger.error("Git CLI not found. Ensure Git is installed and in PATH.")
            diff_text = None
        except Exception as e:
            logger.error("Unexpected error in git diff: %s", str(e))
            diff_text = None
    else:
        diff_text = None

    if diff_text is None:
        logger.debug("Falling back to difflib.unified_diff")
        diff_gen = difflib.unified_diff(
            from_lines, to_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}",
            lineterm="", n=3
        )
        diff_text = "\n".join(diff_gen) or ""
        logger.debug("difflib unified diff output length: %d", len(diff_text))

    # Colorize lines with substring highlighting
    colored_lines = []
    if diff_text:
        lines = diff_text.splitlines()
        for i, line in enumerate(lines):
            prev_line = lines[i - 1] if i > 0 else ""
            colored_lines.append(
                colorize_diff_line_with_substrings(
                    line,
                    is_addition=line.startswith('+') and not line.startswith('+++'),
                    is_deletion=line.startswith('-') and not line.startswith('---'),
                    prev_line=prev_line
                )
            )
    colored_diff = "\n".join(colored_lines) if colored_lines else "No differences."

    # Parse diff for structured changes
    if diff_text:
        current_from_start = 0
        current_to_start = 0
        lines = diff_text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('@@'):
                parts = line.split()
                if len(parts) >= 3:
                    from_range = parts[1].lstrip('-').split(',')
                    to_range = parts[2].lstrip('+').split(',')
                    from_start = int(from_range[0]) - 1
                    from_len = int(from_range[1]) if len(from_range) > 1 else 1
                    to_start = int(to_range[0]) - 1
                    to_len = int(to_range[1]) if len(to_range) > 1 else 1
                    current_from_start = from_start
                    current_to_start = to_start
                    i += 1
                    from_text = []
                    to_text = []
                    while i < len(lines) and not lines[i].startswith('@@'):
                        if lines[i].startswith('-'):
                            from_text.append(lines[i][1:])
                        elif lines[i].startswith('+'):
                            to_text.append(lines[i][1:])
                        else:
                            if from_text or to_text:
                                if from_text and to_text:
                                    changes.append({
                                        "status": "change",
                                        "from_range": {
                                            "start": current_from_start,
                                            "end": current_from_start + len(from_text),
                                            "text": "\n".join(from_text)
                                        },
                                        "to_range": {
                                            "start": current_to_start,
                                            "end": current_to_start + len(to_text),
                                            "text": "\n".join(to_text)
                                        }
                                    })
                                elif from_text:
                                    changes.append({
                                        "status": "remove",
                                        "from_range": {
                                            "start": current_from_start,
                                            "end": current_from_start + len(from_text),
                                            "text": "\n".join(from_text)
                                        },
                                        "to_range": {
                                            "start": current_to_start,
                                            "end": current_to_start,
                                            "text": ""
                                        }
                                    })
                                elif to_text:
                                    changes.append({
                                        "status": "add",
                                        "from_range": {
                                            "start": current_from_start,
                                            "end": current_from_start,
                                            "text": ""
                                        },
                                        "to_range": {
                                            "start": current_to_start,
                                            "end": current_to_start + len(to_text),
                                            "text": "\n".join(to_text)
                                        }
                                    })
                                from_text = []
                                to_text = []
                            current_from_start += 1
                            current_to_start += 1
                        i += 1
                    if from_text or to_text:
                        if from_text and to_text:
                            changes.append({
                                "status": "change",
                                "from_range": {
                                    "start": current_from_start,
                                    "end": current_from_start + len(from_text),
                                    "text": "\n".join(from_text)
                                },
                                "to_range": {
                                    "start": current_to_start,
                                    "end": current_to_start + len(to_text),
                                    "text": "\n".join(to_text)
                                }
                            })
                        elif from_text:
                            changes.append({
                                "status": "remove",
                                "from_range": {
                                    "start": current_from_start,
                                    "end": current_from_start + len(from_text),
                                    "text": "\n".join(from_text)
                                },
                                "to_range": {
                                    "start": current_to_start,
                                    "end": current_to_start,
                                    "text": ""
                                }
                            })
                        elif to_text:
                            changes.append({
                                "status": "add",
                                "from_range": {
                                    "start": current_from_start,
                                    "end": current_from_start,
                                    "text": ""
                                },
                                "to_range": {
                                    "start": current_to_start,
                                    "end": current_to_start + len(to_text),
                                    "text": "\n".join(to_text)
                                }
                            })
                continue
            i += 1

    return changes, colored_diff

def generate_summary_data(
    input_text: str,
    chunks_regular: List[str],
    chunks_fast: List[str]
) -> SummaryData:
    """Generate a summary of the comparison between chunking methods."""
    from datetime import datetime
    differences = []
    for i, (reg, fast) in enumerate(zip(chunks_regular, chunks_fast)):
        if reg != fast:
            reg_lines = reg.splitlines()
            fast_lines = fast.splitlines()
            unified_diff, _ = generate_unified_diff_summary(reg_lines, fast_lines, f"chunk_{i}")
            differences.append({
                "chunk_index": i,
                "regular_content": reg,
                "fast_content": fast,
                "unified_diff": unified_diff
            })
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "input_length": len(input_text),
        "chunk_count": len(chunks_regular),
        "methods_match": chunks_regular == chunks_fast,
        "differences": differences
    }

if __name__ == "__main__":
    sample = "[ Resources ](/r/LangChain/?f=flair_name%3A%22Resources%22)\nI really liked this idea of evaluating different RAG strategies. This simple project is amazing and can be useful to the community here. You can have your custom data evaluate different RAG strategies and finally can see which one works best. Try and let me know what you guys think: [ https://www.ragarena.com/ ](https://www.ragarena.com/)\nPublic\nAnyone can view, post, and comment to thisck-the-right-method-and-why-your-ai-s-success-2cedcda99f8a&user=Mehulpratapsingh&userId=99320eff683e&source=---header_actions--2cedcda99f8a---------------------clap_footer------------------)\n\\--\n[ ](/ leverages both visual and textual information.\n4. SafeRAG: This paper talks covers the benchmark designed to evaluate the security vulnerabilities of RAG systems against adversarial attacks.\n5. Agentic RAG : This paper covers Agentic RAG, which is the fusion of RAG with agents, improving the retrieval process with decision-making and reasoning capabilities.\n6. TrustRAG: This is another paper that covers a security-focused framework designed to protect Retrieval-Augmented Generation (RAG) systems from corpus poisoning attacks.\n7. Enhancing RAG: Best Practices: This study explores key design factors influencing RAG systems, including query expansion, retrieval strategies, and In-Context Learning.\n8. Chain of Retrieval Augmented Generation: This paper covers the CoRG technique that improves RAG by iteratively retrieving and reasoning over the information before generating an answer.\n9. Fact, Fetch and Reason: This paper talks about a high-quality evaluation dataset called FRAMES, designed to evaluate LLMs' factuality, retrieval, and reasoning in end-to-end RAG scenarios.\n10. LONG 2 RAG: LONG 2 RAG is a new benchmark designed to evaluate RAG systems on long-context retrieval and long-form response generation."

    # Log the input text
    logger.info("Input text:\n%s", sample)
    
    # Process with chunk_texts_with_data
    logger.info("Processing with chunk_texts_with_data...")
    result: List[ChunkDict] = chunk_texts_with_data(sample, chunk_size=64,
                                  chunk_overlap=32, model="embeddinggemma")
    save_file(result, f"{OUTPUT_DIR}/result.json")

    # Process with chunk_texts_fast
    logger.info("Processing with chunk_texts_fast...")
    result_fast: List[str] = chunk_texts_fast(sample, chunk_size=64,
                                            chunk_overlap=32, model="embeddinggemma")
    save_file(result_fast, f"{OUTPUT_DIR}/result_fast.json")

    # Compare and log differences using difflib.HtmlDiff + colored unified summary
    logger.info("Comparing chunk contents between methods...")
    chunks_regular = [c["content"] for c in result]
    chunks_fast = [c for c in result_fast]
    
    # Generate and save summary data
    summary_data = generate_summary_data(sample, chunks_regular, chunks_fast)
    save_file(summary_data, f"{OUTPUT_DIR}/summary.json")
    logger.info(f"Saved comparison summary to {OUTPUT_DIR}/summary.json")

    if chunks_regular == chunks_fast:
        logger.info("All chunk contents match between methods.")
    else:
        logger.warning("Chunk contents differ between methods! Generating HTML diff with colored summary...")
        differ = difflib.HtmlDiff(wrapcolumn=80)
        for i, (reg, fast) in enumerate(zip(chunks_regular, chunks_fast)):
            if reg != fast:
                reg_str = dict_to_string(reg)
                fast_str = dict_to_string(fast)
                reg_lines = reg_str.splitlines()
                fast_lines = fast_str.splitlines()
                
                # Generate side-by-side HTML diff
                html_table = differ.make_file(
                    reg_lines, fast_lines,
                    fromdesc=f"Chunk {i} (Regular)", todesc=f"Chunk {i} (Fast)",
                    context=True, numlines=3
                )
                
                # Generate Git-like unified diff summary with substring coloring
                _, unified_summary = generate_unified_diff_summary(reg_lines, fast_lines, f"chunk_{i}")
                
                # Combine: HTML table + colored unified summary with styles
                html_diff = f"""
                {html_table}
                <hr style="border: 1px solid #ccc; margin: 20px 0;">
                <h3>Git Diff-Style Unified Summary</h3>
                <pre style="background: #f8f8f8; padding: 15px; border: 1px solid #ddd; overflow-x: auto; font-family: monospace; font-size: 14px; line-height: 1.5; white-space: pre-wrap;">
{unified_summary}
                </pre>
                """
                
                # Log and save HTML diff
                logger.warning(f"Difference in chunk {i}:\n{html_diff}")
                diff_file = f"{OUTPUT_DIR}/diff_chunk_{i}.html"
                with open(diff_file, 'w', encoding='utf-8') as f:
                    f.write(html_diff)
                logger.info(f"Saved HTML diff for chunk {i} to {diff_file}")

    # Assert that all chunk dictionaries (without _id keys) are the same
    assert chunks_regular == chunks_fast, "Chunk contents differ between methods"
    logger.info("Assertion passed: All chunk contents are identical.")
