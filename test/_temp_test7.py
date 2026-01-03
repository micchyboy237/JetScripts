import audiodiff

def are_perceptually_equal(bytes1: bytes, bytes2: bytes) -> bool:
    """True if audio streams are perceptually identical (after decoding)."""
    # audiodiff works on file paths; write to temp files for bytes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as f1, tempfile.NamedTemporaryFile(suffix=".wav") as f2:
        f1.write(bytes1)
        f2.write(bytes2)
        f1.flush()
        f2.flush()
        return audiodiff.audio_equal(f1.name, f2.name)

# Example
# print("[green]Equal[/green]" if are_perceptually_equal(bytes1, bytes2) else "[red]Different[/red]")