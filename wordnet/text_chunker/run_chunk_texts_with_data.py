import os
import logging
import difflib
import json
from typing import TypedDict, List
from jet.file.utils import save_file
from jet.wordnet.text_chunker import chunk_texts_with_data, chunk_texts_with_data_fast

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

sample = "[ Resources ](/r/LangChain/?f=flair_name%3A%22Resources%22)\nI really liked this idea of evaluating different RAG strategies. This simple project is amazing and can be useful to the community here. You can have your custom data evaluate different RAG strategies and finally can see which one works best. Try and let me know what you guys think: [ https://www.ragarena.com/ ](https://www.ragarena.com/)\nPublic\nAnyone can view, post, and comment to thisck-the-right-method-and-why-your-ai-s-success-2cedcda99f8a&user=Mehulpratapsingh&userId=99320eff683e&source=---header_actions--2cedcda99f8a---------------------clap_footer------------------)\n\\--\n[ ](/ leverages both visual and textual information.\n4. SafeRAG: This paper talks covers the benchmark designed to evaluate the security vulnerabilities of RAG systems against adversarial attacks.\n5. Agentic RAG : This paper covers Agentic RAG, which is the fusion of RAG with agents, improving the retrieval process with decision-making and reasoning capabilities.\n6. TrustRAG: This is another paper that covers a security-focused framework designed to protect Retrieval-Augmented Generation (RAG) systems from corpus poisoning attacks.\n7. Enhancing RAG: Best Practices: This study explores key design factors influencing RAG systems, including query expansion, retrieval strategies, and In-Context Learning.\n8. Chain of Retrieval Augmented Generation: This paper covers the CoRG technique that improves RAG by iteratively retrieving and reasoning over the information before generating an answer.\n9. Fact, Fetch and Reason: This paper talks about a high-quality evaluation dataset called FRAMES, designed to evaluate LLMs' factuality, retrieval, and reasoning in end-to-end RAG scenarios.\n10. LONG 2 RAG: LONG 2 RAG is a new benchmark designed to evaluate RAG systems on long-context retrieval and long-form response generation."

class ChunkDict(TypedDict):
    content: str
    # Add other expected keys as needed; filtered dynamically

def filter_dict(d: dict) -> dict:
    """Filter out keys ending with '_id' or equal to 'id' from a dictionary."""
    return {k: v for k, v in d.items() if not (k.endswith('_id') or k == 'id')}

def dict_to_string(d: dict) -> str:
    """Convert a dictionary to a formatted string for diff comparison."""
    return json.dumps(d, indent=2, sort_keys=True)

def html_escape(text: str) -> str:
    """Escape HTML special characters for safe <pre> display."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

def generate_unified_diff_summary(
    from_lines: List[str], to_lines: List[str], path: str = "chunk"
) -> str:
    """
    Generate a Git-like unified diff summary (stats + hunks) for two sets of lines.
    Falls back to difflib's unified_diff if GitPython unavailable or Git CLI fails.
    """
    if GITPYTHON_AVAILABLE:
        # Use GitPython for authentic Git diff (treat lines as file content)
        from_content = "\n".join(from_lines) + "\n"
        to_content = "\n".join(to_lines) + "\n"
        # Combine inputs with separator as a single string
        input_content = f"{from_content}\n---\n{to_content}"
        # Simulate Git diff via subprocess for unified output
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
            logger.debug("Git diff output length: %d", len(result.stdout))
            return result.stdout if result.stdout.strip() else "No differences."
        except subprocess.CalledProcessError as e:
            logger.error("Git diff failed: %s", e.stderr)
            # Fallback to difflib
        except FileNotFoundError:
            logger.error("Git CLI not found. Ensure Git is installed and in PATH.")
            # Fallback to difflib
        except Exception as e:
            logger.error("Unexpected error in git diff: %s", str(e))
            # Fallback to difflib

    # Fallback to difflib (built-in, no deps)
    logger.debug("Falling back to difflib.unified_diff")
    diff_gen = difflib.unified_diff(
        from_lines, to_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        lineterm="", n=3  # 3 lines of context, like git diff -U3
    )
    unified_diff = "\n".join(diff_gen)
    logger.debug("difflib unified diff output length: %d", len(unified_diff))
    return unified_diff or "No differences."

if __name__ == "__main__":
    # Log the input text
    logger.info("Input text:\n%s", sample)
    
    # Process with chunk_texts_with_data
    logger.info("Processing with chunk_texts_with_data...")
    result: List[ChunkDict] = chunk_texts_with_data(sample, chunk_size=64,
                                  chunk_overlap=32, model="embeddinggemma")
    logger.info("chunk_texts_with_data results:")
    for i, chunk in enumerate(result):
        filtered_chunk = filter_dict(chunk)
        logger.info("Chunk %d: %s", i + 1, dict_to_string(filtered_chunk))
    save_file(result, f"{OUTPUT_DIR}/result.json")

    # Process with chunk_texts_with_data_fast
    logger.info("Processing with chunk_texts_with_data_fast...")
    result_fast: List[ChunkDict] = chunk_texts_with_data_fast(sample, chunk_size=64,
                                            chunk_overlap=32, model="embeddinggemma")
    logger.info("chunk_texts_with_data_fast results:")
    for i, chunk in enumerate(result_fast):
        filtered_chunk = filter_dict(chunk)
        logger.info("Chunk %d: %s", i + 1, dict_to_string(filtered_chunk))
    save_file(result_fast, f"{OUTPUT_DIR}/result_fast.json")

    # Compare and log differences using difflib.HtmlDiff + unified summary
    logger.info("Comparing chunk contents between methods...")
    chunks_regular = [filter_dict(c) for c in result]
    chunks_fast = [filter_dict(c) for c in result_fast]
    if chunks_regular == chunks_fast:
        logger.info("All chunk contents match between methods.")
    else:
        logger.warning("Chunk contents differ between methods! Generating HTML diff with summary...")
        differ = difflib.HtmlDiff(wrapcolumn=80)  # Wrap lines for readability
        for i, (reg, fast) in enumerate(zip(chunks_regular, chunks_fast)):
            if reg != fast:
                # Convert filtered dicts to strings for diff
                reg_str = dict_to_string(reg)
                fast_str = dict_to_string(fast)
                reg_lines = reg_str.splitlines()
                fast_lines = fast_str.splitlines()
                
                # Generate side-by-side HTML diff
                html_table = differ.make_file(
                    reg_lines, fast_lines,
                    fromdesc=f"Chunk {i+1} (Regular)", todesc=f"Chunk {i+1} (Fast)",
                    context=True, numlines=3
                )
                
                # Generate Git-like unified diff summary
                unified_summary = generate_unified_diff_summary(reg_lines, fast_lines, f"chunk_{i+1}")
                
                # Combine: HTML table + unified summary below
                html_diff = f"""
                {html_table}
                <hr style="border: 1px solid #ccc; margin: 20px 0;">
                <h3>Git Diff-Style Unified Summary</h3>
                <pre style="background: #f8f8f8; padding: 10px; border: 1px solid #ddd; overflow-x: auto; font-family: monospace;">
{html_escape(unified_summary)}
                </pre>
                """
                
                logger.warning(f"Difference in chunk {i+1}:\n{html_diff}")
                # Save HTML diff to file for browser viewing
                diff_file = f"{OUTPUT_DIR}/diff_chunk_{i+1}.html"
                with open(diff_file, 'w', encoding='utf-8') as f:
                    f.write(html_diff)
                logger.info(f"Saved HTML diff for chunk {i+1} to {diff_file}")

    # Assert that all chunk dictionaries (without _id keys) are the same
    assert chunks_regular == chunks_fast, "Chunk contents differ between methods"
    logger.info("Assertion passed: All chunk contents are identical.")