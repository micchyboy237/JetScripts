from chunker import Chunker

def main():
    """
    A main function for the chunker class.

    This function tests the chunker class with different usage examples.

    Usage Examples:
    1. Test the chunker class with a simple text data.
    2. Test the chunker class with a text data from a PDF file.
    3. Test the chunker class with a text data from a PDF file with some metadata.
    4. Test the chunker class with a text data from a PDF file with some metadata and an overlap.
    """

    # Test the chunker class with a simple text data
    chunker = Chunker()
    text_data = [
        {"content": "This is a simple text data", "metadata": {"author": "John Doe", "date": "2022-01-01"}},
        {"content": "This is another simple text data", "metadata": {"author": "Jane Doe", "date": "2022-01-02"}},
        {"content": "This is yet another simple text data", "metadata": {"author": "Bob Smith", "date": "2022-01-03"}},
    ]
    chunked_data = chunker.chunk_text(text_data, 100, 5)
    print(chunked_data)

    # Test the chunker class with a text data from a PDF file
    pdf_file = "example.pdf"
    text_data = []
    with open(pdf_file, "rb") as f:
        for line in f:
            text_data.append({"content": line.decode("utf-8"), "metadata": {"author": "John Doe", "date": "2022-01-01"}}))
    chunker = Chunker()
    chunked_data = chunker.chunk_text(text_data, 100, 5)
    print(chunked_data)

    # Test the chunker class with a text data from a PDF file with some metadata
    pdf_file = "example.pdf"
    text_data = []
    with open(pdf_file, "rb") as f:
        for line in f:
            text_data.append({"content": line.decode("utf-8"), "metadata": {"author": "John Doe", "date": "2022-01-01"}}))
    chunker = Chunker()
    chunked_data = chunker.chunk_text(text_data, 100, 5)
    print(chunked_data)

    # Test the chunker class with a text data from a PDF file with some metadata and an overlap
    pdf_file = "example.pdf"
    text_data = []
    with open(pdf_file, "rb") as f:
        for line in f:
            text_data.append({"content": line.decode("utf-8"), "metadata": {"author": "John Doe", "date": "2022-01-01"}}))
    chunker = Chunker()
    chunked_data = chunker.chunk_text(text_data, 100, 5)
    print(chunked_data)

    # Run all examples
    print("Running all examples...")
    chunker = Chunker()
    chunked_data = chunker.chunk_text(text_data, 100, 5)
    print(chunked_data)

if __name__ == "__main__":
    main()