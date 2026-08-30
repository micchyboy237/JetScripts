import json
import logging
import shutil
import time
from pathlib import Path

from rich.console import Console
from unstructured.partition.html.transformations import (
    ontology_to_unstructured_elements,
    parse_html_to_ontology,
)
from unstructured.staging.base import elements_from_json

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = OUTPUT_DIR / "processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

html_file_path = Path(
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/unstructured/test_unstructured/documents/html_files/example.html"
)
json_file_path = Path(
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/unstructured/test_unstructured/documents/unstructured_json_output/example.json"
)


# -----------------------------------------------------------------------------
# Main Processing Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    start_time = time.perf_counter()
    saved_files: list[Path] = []

    logger.info("Starting HTML-to-Unstructured conversion pipeline")
    logger.info("Output directory: %s", OUTPUT_DIR)

    # --- Load inputs ---------------------------------------------------------
    logger.info("Loading HTML from %s", html_file_path.name)
    html_code = html_file_path.read_text(encoding="utf-8")
    logger.info("HTML loaded (%d characters)", len(html_code))

    logger.info("Loading expected JSON elements from %s", json_file_path.name)
    expected_json_elements = elements_from_json(str(json_file_path))
    logger.info("Expected elements loaded: %d items", len(expected_json_elements))

    expected_out = OUTPUT_DIR / "expected_elements.json"
    expected_out.write_text(
        json.dumps([e.to_dict() for e in expected_json_elements], indent=2),
        encoding="utf-8",
    )
    saved_files.append(expected_out)
    logger.info("Saved expected elements → %s", expected_out.name)

    # --- Parse HTML to Ontology ----------------------------------------------
    logger.info("Parsing HTML to ontology…")
    t0 = time.perf_counter()
    ontology = parse_html_to_ontology(html_code)
    logger.info("Ontology parsed in %.3fs", time.perf_counter() - t0)

    ontology_out = OUTPUT_DIR / "ontology.json"
    ontology_out.write_text(
        json.dumps(ontology.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )
    saved_files.append(ontology_out)
    logger.info("Saved ontology → %s", ontology_out.name)

    # --- Convert Ontology to Unstructured Elements ---------------------------
    logger.info("Converting ontology to Unstructured elements…")
    t0 = time.perf_counter()
    unstructured_elements = ontology_to_unstructured_elements(ontology)
    logger.info(
        "Conversion complete in %.3fs — %d elements produced",
        time.perf_counter() - t0,
        len(unstructured_elements),
    )

    result_out = OUTPUT_DIR / "unstructured_elements.json"
    result_out.write_text(
        json.dumps([e.to_dict() for e in unstructured_elements], indent=2),
        encoding="utf-8",
    )
    saved_files.append(result_out)
    logger.info("Saved unstructured elements → %s", result_out.name)

    # Always include the log file itself as a saved artifact
    saved_files.append(log_file)

    # --- Summary with Resource Links -----------------------------------------
    elapsed = time.perf_counter() - start_time
    logger.info("Pipeline finished in %.3fs", elapsed)

    console.print()
    console.rule("[bold green]✅ Pipeline Complete[/bold green]")
    console.print(f"[dim]Total time: {elapsed:.3f}s[/dim]\n")
    console.print("[bold]Saved resources:[/bold]")
    for filepath in sorted(saved_files):
        # file:// URI makes the link clickable in most terminals/IDEs
        console.print(f"  📄 [link=file://{filepath}]{filepath.name}[/link]")
    console.print()


if __name__ == "__main__":
    main()
