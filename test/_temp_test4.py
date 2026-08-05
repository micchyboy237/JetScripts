from unstructured.partition.pdf import partition_pdf

# hi_res strategy uses layout detection + OCR for complex PDFs
elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,  # Save embedded images as separate elements
    extract_table_structure=True,  # Preserve table rows/columns
    languages=["eng"],
)

# Inspect element types and content
for el in elements[:5]:
    print(f"[{el.category}] {str(el)[:120]}")
