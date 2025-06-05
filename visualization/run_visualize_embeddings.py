from visualization.visualize_embeddings import visualize_embeddings


if __name__ == "__main__":
    text = "Apple is based in California."
    chart_config, tagged_entities = visualize_embeddings(text)
    print(f"Tagged entities: {tagged_entities}")
