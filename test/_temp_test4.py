# scripts/merge_context_files.py

from pathlib import Path


def merge_context_md_files(base_dir: Path, output_file: Path) -> None:
    """
    Recursively finds all 'context.md' files and writes their contents
    into a single markdown file.

    Args:
        base_dir (Path): Directory to start search from.
        output_file (Path): Output file path.
    """
    context_files = list(base_dir.rglob("context.md"))

    with output_file.open("w", encoding="utf-8") as out_f:
        for path in sorted(context_files):
            out_f.write(f"# {path.relative_to(base_dir)}\n\n")
            out_f.write(path.read_text(encoding="utf-8"))
            out_f.write("\n\n---\n\n")


if __name__ == "__main__":
    import sys

    try:
        base_path = Path(sys.argv[1]).expanduser().resolve()
        if not base_path.is_dir():
            raise ValueError(f"Provided path is not a directory: {base_path}")

        output_path = base_path / "searched_isekai_anime.md"
        merge_context_md_files(base_path, output_path)
        print(f"✅ Merged content written to: {output_path}")

    except IndexError:
        print("Usage: python scripts/merge_context_files.py <base_dir>")
    except Exception as e:
        print(f"❌ Error: {e}")
