import os
import shutil
from pathlib import Path
from typing import List

from jet.file.utils import save_data, save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs_ai_llm_python
from jet.wordnet.analyzers.analyze_pos_tags import analyze_pos_tags
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main_pos_tagger(texts: List[str], output_dir: str) -> None:
    """
    Main function to perform POS tagging analysis on texts.

    Args:
        texts: List of text strings to analyze
        output_dir: Directory to save the results
    """
    console.print(
        "\n[bold magenta]╔════════════════════════════════════════╗[/bold magenta]"
    )
    console.print(
        "[bold magenta]║   POS Tag Analysis - AI Jobs          ║[/bold magenta]"
    )
    console.print(
        "[bold magenta]╚════════════════════════════════════════╝[/bold magenta]\n"
    )

    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Number of texts:[/cyan] {len(texts)}")

    # Save raw texts
    save_file(texts, f"{output_dir}/texts.json")
    console.print(f"[green]✓ Saved raw texts to texts.json[/green]\n")

    # Prepare texts with language info
    texts_with_lang = [{"text": t, "lang": "en"} for t in texts]

    includes_pos = ["PROPN", "NOUN"]
    excludes_pos = ["[BOS]", "[EOS]"]
    top_n = 50

    # Analysis 1: Single n=2 bigram analysis
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]Analysis 1: Single n=2 Bigram Analysis[/bold yellow]")
    console.print("=" * 60)

    results_n2 = analyze_pos_tags(
        texts_with_lang,
        n=2,
        words_only=True,
        top_n=top_n,
        includes_pos=includes_pos,
    )

    results_dir = Path(output_dir) / "single_n2_bigrams"
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[cyan]Saving {len(results_n2)} result files to:[/cyan] {results_dir}"
    )
    for filename, data in results_n2.items():
        # Convert Path to string for save_data
        save_path = str(results_dir / f"{filename}.json")
        data_size = len(data) if isinstance(data, dict) else "scalar"
        console.print(f"  Saving: [dim]{filename}.json[/dim] ({data_size} items)")
        save_data(save_path, data)

    console.print(
        f"[green]✓ Saved {len(results_n2)} files to single_n2_bigrams/[/green]"
    )

    # Analysis 2: Multi-ngram range (1-3) with includes_pos
    console.print("\n" + "=" * 60)
    console.print(
        "[bold yellow]Analysis 2: Ngram Range (1-3) with includes_pos=[PROPN, NOUN][/bold yellow]"
    )
    console.print("=" * 60)

    results_range_includes = analyze_pos_tags(
        texts_with_lang,
        ngram_range=(1, 3),
        words_only=True,
        top_n=top_n,
        includes_pos=includes_pos,
    )

    results_dir = Path(output_dir) / "range_1_3_includes"
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[cyan]Saving {len(results_range_includes)} result files to:[/cyan] {results_dir}"
    )
    for filename, data in results_range_includes.items():
        # Convert Path to string for save_data
        save_path = str(results_dir / f"{filename}.json")
        data_size = len(data) if isinstance(data, dict) else "scalar"
        console.print(f"  Saving: [dim]{filename}.json[/dim] ({data_size} items)")
        save_data(save_path, data)

    console.print(
        f"[green]✓ Saved {len(results_range_includes)} files to range_1_3_includes/[/green]"
    )

    # Analysis 3: Multi-ngram range (1-3) with excludes_pos
    console.print("\n" + "=" * 60)
    console.print(
        "[bold yellow]Analysis 3: Ngram Range (1-3) with excludes_pos=[BOS], [EOS][/bold yellow]"
    )
    console.print("=" * 60)

    results_range_excludes = analyze_pos_tags(
        texts_with_lang,
        ngram_range=(1, 3),
        words_only=True,
        top_n=top_n,
        includes_pos=includes_pos,
        excludes_pos=excludes_pos,
    )

    results_dir = Path(output_dir) / "range_1_3_excludes"
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[cyan]Saving {len(results_range_excludes)} result files to:[/cyan] {results_dir}"
    )
    for filename, data in results_range_excludes.items():
        # Convert Path to string for save_data
        save_path = str(results_dir / f"{filename}.json")
        data_size = len(data) if isinstance(data, dict) else "scalar"
        console.print(f"  Saving: [dim]{filename}.json[/dim] ({data_size} items)")
        save_data(save_path, data)

    console.print(
        f"[green]✓ Saved {len(results_range_excludes)} files to range_1_3_excludes/[/green]"
    )

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]✓ All analyses complete![/bold green]")
    console.print("=" * 60)
    console.print(f"\n[cyan]Results saved to:[/cyan] {output_dir}")
    console.print(f"  - single_n2_bigrams/ ({len(results_n2)} files)")
    console.print(f"  - range_1_3_includes/ ({len(results_range_includes)} files)")
    console.print(f"  - range_1_3_excludes/ ({len(results_range_excludes)} files)")


if __name__ == "__main__":
    console.print("[dim]Starting script...[/dim]")

    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0],
    )

    console.print(f"[cyan]Loading sample jobs...[/cyan]")
    texts = load_sample_jobs_ai_llm_python()

    sub_output_dir = f"{output_dir}/jobs"

    try:
        main_pos_tagger(texts, sub_output_dir)
        console.print("\n[bold green]✓ Script completed successfully![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error occurred:[/bold red] {e}")
        console.print_exception()
        raise
