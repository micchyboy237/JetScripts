import shutil
from pathlib import Path
from typing import Any

from jet.file.utils import save_file
from jet.vectors.ner import (
    Entity,
    extract_entities,
    extract_entities_from_text,
    load_nlp_pipeline,
)
from rich.console import Console
from rich.progress import track
from rich.table import Table
from shared.data_types.job import JobEntity

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

DEFAULT_GLNER_MODEL = "urchade/gliner_large-v2.1"
DEFAULT_CHUNK_SIZE = 512

# Assuming these are in the same file or imported:
# from your_module import (
#     load_nlp_pipeline, extract_entities, Entity
# )

# For self-contained example — paste your functions here if needed

# ────────────────────────────────────────────────
#   Example job-related labels (zero-shot — GLiNER handles them)
# ────────────────────────────────────────────────
JOB_LABELS = [
    "job title",
    "posted date",
    "company name",
    "job location",
    "salary range",
    "experience level",
    "employment type",
    "work schedule",
    "required skills",
    "used technologies",
    "programming languages",
    "key responsibilities",
    "requirements qualifications",
    "employee benefits",
]

# ────────────────────────────────────────────────
#   Simulated scraped job texts (replace with your real data)
# ────────────────────────────────────────────────
SAMPLE_JOB_POSTINGS = [
    """
    Senior Python Backend Developer
    Posted: October 15, 2025
    TechCorp Inc. | Remote (USA)
    $140,000 - $180,000 / year + 15% bonus
    Full-time

    We are looking for a Senior Python Backend Developer with 5+ years of professional experience.
    Required skills: API design, system architecture, unit testing.
    Technologies: Python, FastAPI, PostgreSQL, Docker, AWS, Redis, Kafka.
    Programming languages: Python, Go (nice to have).
    Responsibilities: Design scalable microservices, mentor junior developers, conduct code reviews, improve CI/CD pipelines.
    Qualifications: BS/MS in Computer Science or equivalent experience, strong knowledge of distributed systems.
    Benefits: Health/dental/vision insurance, 401k match up to 6%, unlimited PTO, remote work stipend, annual learning budget.
    """,
    """
    Data Scientist - Machine Learning Engineer
    Posted on: November 3, 2025
    AI Innovations Ltd | New York, NY / Hybrid (3 days office)
    $120,000–$160,000 base + performance bonus + equity

    Join our team to build cutting-edge ML models and recommendation systems.
    Skills: Python, statistical modeling, A/B testing, experiment design.
    Technologies & Tools: PyTorch, TensorFlow, scikit-learn, SQL, pandas, Spark, MLflow.
    Programming languages: Python, R, SQL.
    Experience level: 3–7 years in industry ML roles.
    Responsibilities: Develop and deploy production ML models, run A/B tests, collaborate with product & engineering teams.
    Qualifications: Master's/PhD in CS, Statistics, or related field preferred; experience shipping models to production.
    Benefits: Comprehensive health coverage, stock options, flexible hours, professional development stipend.
    """,
    """
    Frontend Engineer (React & TypeScript)
    Posted Date: December 10, 2025
    StartupX | San Francisco, CA (On-site preferred, hybrid possible)
    $130k–$170k + significant equity + signing bonus

    We're hiring a Frontend Engineer!
    Must have skills: modern JavaScript, responsive design, state management.
    Technologies: React, TypeScript, Redux Toolkit, Tailwind CSS, Jest, Vite, Next.js (bonus).
    Programming languages: JavaScript/TypeScript.
    Experience: 2–5 years building complex web applications.
    Responsibilities: Build pixel-perfect responsive UIs, optimize frontend performance, collaborate with designers & backend engineers.
    Qualifications: Strong portfolio of shipped products, experience with modern frontend tooling.
    Benefits: Medical/dental/vision, 401k, equity package, unlimited PTO, home office setup allowance.
    """,
]


# ────────────────────────────────────────────────
#   Main extraction & display logic
# ────────────────────────────────────────────────
def extract_jobs_from_texts(
    texts: list[str],
    labels: list[str] = JOB_LABELS,
    style: str = "ent",
    model: str = DEFAULT_GLNER_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    threshold: float = 0.3,  # optional filter — tune as needed
) -> list[dict[str, Any]]:
    """Process multiple job postings and return structured results."""
    console = Console()

    with console.status("[bold cyan]Loading GLiNER + spaCy pipeline..."):
        nlp = load_nlp_pipeline(
            labels=labels,
            style=style,
            model=model,
            chunk_size=chunk_size,
        )
    console.log("[green]Pipeline ready[/green]")

    extracted_entities = []
    extracted_jobs = []

    for text in track(texts, description="Extracting entities from jobs..."):
        # Optional: clean text more if needed (remove boilerplate, etc.)
        entities: list[Entity] = extract_entities(nlp, text, threshold=threshold)
        job_entities: JobEntity = extract_entities_from_text(
            nlp, text, threshold=threshold
        )

        # Optional: filter low-confidence (if your extract function doesn't already)
        # But since get_unique_entities keeps highest score, usually ok

        entities_record = {
            "snippet": text[:180].strip() + "...",
            "entities": entities,
        }
        job_entities_record = {
            "snippet": text[:180].strip() + "...",
            "extracted": job_entities,
        }
        extracted_entities.append(entities_record)
        extracted_jobs.append(job_entities_record)

    return extracted_entities, extracted_jobs


def display_extracted_jobs(jobs: list[dict[str, Any]]):
    """Pretty-print results using rich."""
    console = Console()

    table = Table(
        title="Extracted Job Postings", show_header=True, header_style="bold magenta"
    )
    table.add_column("Job Snippet", style="dim", width=40)
    table.add_column("Key Entities", style="cyan")

    for job in jobs:
        snippet = job["snippet"]
        entities_lines = []

        for key, values in job["extracted"].items():
            if values:  # skip empty
                nice_key = key.replace("_", " ").title()
                entities_lines.append(
                    f"[bold]{nice_key}:[/bold] {', '.join(values[:4])}"
                )

        entities_text = (
            "\n".join(entities_lines) or "[italic]No entities found[/italic]"
        )

        table.add_row(snippet, entities_text)

    console.print(table)


# ────────────────────────────────────────────────
#   Run example
# ────────────────────────────────────────────────
if __name__ == "__main__":
    console = Console()
    console.rule("Job Postings Entity Extraction Demo")

    extracted_entities, extracted_jobs = extract_jobs_from_texts(
        SAMPLE_JOB_POSTINGS,
        labels=JOB_LABELS,
    )

    display_extracted_jobs(extracted_jobs)

    save_file(extracted_entities, f"{OUTPUT_DIR}/entities.json")
    save_file(extracted_jobs, f"{OUTPUT_DIR}/jobs.json")

    console.rule("Done")
