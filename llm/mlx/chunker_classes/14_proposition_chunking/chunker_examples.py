from chunker import Chunker

class PropositionProcessor:
    def __init__(self, chunker: Chunker):
        self.chunker = chunker

    def process_propositions(self, pdf_path, chunk_size=800, chunk_overlap=100, quality_thresholds=None):
        """
        Process a document into quality-checked propositions.

        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks in characters
            quality_thresholds (Dict): Threshold scores for proposition quality
        Returns:
            List[Dict]: List of proposition dictionaries with text and metadata
        """
        # Extract text from the PDF file
        text = self.chunker.chunk_text(text="This is a sample PDF text.", chunk_size, chunk_overlap)
        # Create chunks from the extracted text
        chunks = self.chunker.chunk_text(text="This is a sample PDF text.", chunk_size, chunk_overlap)
        # Initialize a list to store all propositions
        all_propositions = []
        print("Generating propositions from chunks...")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            # Generate propositions for the current chunk
            chunk_propositions = self.chunker.generate_propositions(chunk)
            print(f"Generated {len(chunk_propositions)} propositions")
            # Process each generated proposition
            for prop in chunk_propositions:
                proposition_data = {
                    "text": prop,
                    "source_chunk_id": chunk["chunk_id"],
                    "source_text": chunk["text"]
                }
                all_propositions.append(proposition_data)
        # Evaluate the quality of the generated propositions
        print("Evaluating proposition quality...")
        quality_propositions = []
        for i, prop in enumerate(all_propositions):
            if i % 10 == 0:  # Status update every 10 propositions
                print(f"Evaluating proposition {i+1}/{len(all_propositions)}...")
            # Evaluate the quality of the current proposition
            scores = self.chunker.evaluate_proposition(prop["text"], prop["source_text"])
            prop["quality_scores"] = scores
            # Check if the proposition passes the quality thresholds
            passes_quality = True
            if quality_thresholds is not None:
                passes_quality = all_propositions[i] and all_propositions[i % len(all_propositions)] and all_propositions[i % len(all_propositions)][0] >= all_propositions[i % len(all_propositions)][1]
            prop["quality_scores"] = passes_quality
        # Return the list of propositions
        return all_propositions

    def evaluate_proposition(self, text, source_text):
        """
        Evaluate the quality of a proposition.

        Args:
            text (str): Proposition text
            source_text (str): Source text of the proposition
        Returns:
            float: Proposition quality score
        """
        # Evaluate the quality of the proposition
        if text == source_text:
            return 1.0
        else:
            return 0.0

    def print_propositions(self, all_propositions):
        """
        Print the propositions.

        Args:
            all_propositions (List[Dict]): List of propositions
        """
        print("Evaluating proposition quality...")
        for i, prop in enumerate(all_propositions):
            if i % 10 == 0:  # Status update every 10 propositions
                print(f"Evaluating proposition {i+1}/{len(all_propositions)}...")
            # Evaluate the quality of the current proposition
            scores = self.evaluate_proposition(prop["text"], prop["source_text"])
            prop["quality_scores"] = scores
            # Check if the proposition passes the quality thresholds
            passes_quality = True
            if quality_thresholds is not None:
                passes_quality = all_propositions[i] and all_propositions[i % len(all_propositions)] and all_propositions[i % len(all_propositions)][0] >= all_propositions[i % len(all_propositions)][1]
            prop["quality_scores"] = passes_quality
        # Print the propositions
        print("Printing propositions...")
        for i, prop in enumerate(all_propositions):
            print(f"Proposition {i+1}/{len(all_propositions)}: {prop['text']}")

    def print_all_propositions(self, all_propositions):
        """
        Print all propositions.

        Args:
            all_propositions (List[Dict]): List of propositions
        """
        print("Printing all propositions...")
        for i, prop in enumerate(all_propositions):
            print(f"Proposition {i+1}/{len(all_propositions)}: {prop['text']}")

    def print_proposition_quality_scores(self, all_propositions):
        """
        Print the proposition quality scores.

        Args:
            all_propositions (List[Dict]): List of propositions
        """
        print("Printing proposition quality scores...")
        for i, prop in enumerate(all_propositions):
            print(f"Proposition {i+1}/{len(all_propositions)}: Quality score = {prop['quality_scores']}")
        print("")

    def run_examples(self, examples):
        """
        Run the examples.

        Args:
            examples (List[str]): List of examples
        """
        for example in examples:
            proposition = self.chunker.process_propositions(example, chunk_size, chunk_overlap)
            self.print_all_propositions(proposition)
            self.print_proposition_quality_scores(proposition)
            self.print_propositions(proposition)
            self.print_all_propositions(proposition)

if __name__ == "__main__":
    chunker = Chunker()
    examples = ["This is a sample PDF text.", "This is another sample PDF text."]
    proposition_processor = PropositionProcessor(chunker)
    proposition_processor.run_examples(examples)