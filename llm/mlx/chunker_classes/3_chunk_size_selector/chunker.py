```python
import fitz
import os
import numpy as np
import json
from tqdm import tqdm
from openai import OpenAI

class Chunker:
    def __init__(self, model="BAAI/bge-en-icl"):
        self.model = model
        self.openai = OpenAI()

    def chunk_text(self, text, n, overlap):
        """
        Splits text into overlapping chunks.
        Args:
            text (str): The text to be chunked.
            n (int): Number of characters per chunk.
            overlap (int): Overlapping characters between chunks.
        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []  # Initialize an empty list to store the chunks
        for i in range(0, len(text), n - overlap):
            # Append a chunk of text from the current index to the index + chunk size
            chunks.append(text[i:i + n])
        return chunks  # Return the list of text chunks

    def create_embeddings(self, texts):
        """
        Generates embeddings for a list of texts.
        Args:
            texts (List[str]): List of input texts.
            model (str): Embedding model.
        Returns:
            List[np.ndarray]: List of numerical embeddings.
        """
        # Create embeddings using the specified model
        response = self.openai.embeddings.create(model=self.model, input=texts)
        # Convert the response to a list of numpy arrays and return
        return [np.array(embedding.embedding) for embedding in response.data]

    def generate_response(self, query, retrieved_chunks, system_prompt, model="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Generates an AI response based on retrieved chunks.
        Args:
            query (str): User query.
            retrieved_chunks (List[str]): List of retrieved text chunks.
            system_prompt (str): System prompt for the AI assistant.
            model (str): AI model.
        Returns:
            str: AI-generated response.
        """
        # Combine retrieved chunks into a single context string
        context = "n".join([f"Context {i+1}:n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
        # Create the user prompt by combining the context and the query
        user_prompt = f"{context}nnQuestion: {query}"

        # Generate the AI response using the specified model
        response = self.openai.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        # Return the content of the AI response
        return response.choices[0].message.content

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            str: Extracted text from the PDF.
        """
        # Open the PDF file
        mypdf = fitz.open(pdf_path)
        all_text = ""  # Initialize an empty string to store the extracted text
        # Iterate through each page in the PDF
        for page in mypdf:
            # Extract text from the current page and add spacing
            all_text += page.get_text("text") + " "
        # Return the extracted text, stripped of leading/trailing whitespace
        return all_text.strip()

    def evaluate_ai_response(self, query, retrieved_chunks, system_prompt, model="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Evaluates the AI response based on the user query and retrieved chunks.
        Args:
            query (str): User query.
            retrieved_chunks (List[str]): List of retrieved text chunks.
            system_prompt (str): System prompt for the AI assistant.
            model (str): AI model.
        Returns:
            float: Evaluation score for the AI response.
        """
        # Define evaluation scoring system constants
        SCORE_FULL = 1.0     # Complete match or fully satisfactory
        SCORE_PARTIAL = 0.5  # Partial match or somewhat satisfactory
        SCORE_NONE = 0.0     # No match or unsatisfactory
        # Generate embeddings for each chunk size
        chunk_embeddings_dict = {size: self.create_embeddings(chunks) for size, chunks in tqdm(self.chunk_text(self.extract_text_from_pdf("data/AI_Information.pdf"), 128, 128).items(), desc="Generating Embeddings")}
        # Generate AI responses for each chunk size
        ai_responses_dict = {size: self.generate_response(query, retrieved_chunks_dict[size], system_prompt, model="meta-llama/Llama-3.2-3B-Instruct") for size in [128, 256, 512]}
        # Print the response for chunk size 256
        print(ai_responses_dict[256])
        # Evaluate the AI response
        # TO DO: Implement evaluation logic
        return SCORE_FULL  # Replace with actual evaluation logic
```