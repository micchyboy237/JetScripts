```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI

class Chunker:
    def __init__(self, model="meta-llama/Llama-3.2-3B-Instruct"):
        self.model = model
        self.client = OpenAI()

    def split_into_chunks(self, sentences, breakpoints):
        """
        Splits sentences into semantic chunks.
        
        Args:
            sentences (List[str]): List of sentences.
            breakpoints (List[int]): Indices where chunking should occur.
        
        Returns:
            List[str]: List of text chunks.
        """
        chunks = []  # Initialize an empty list to store the chunks
        start = 0  # Initialize the start index
        # Iterate through each breakpoint to create chunks
        for bp in breakpoints:
            # Append the chunk of sentences from start to the current breakpoint
            chunks.append(". ".join(sentences[start:bp + 1]) + ".")  
            start = bp + 1  # Update the start index to the next sentence after the breakpoint
        # Append the remaining sentences as the last chunk
        chunks.append(". ".join(sentences[start:]))
        return chunks  # Return the list of chunks

    def create_embeddings(self, text_chunks):
        """
        Creates embeddings for each text chunk.
        
        Args:
            text_chunks (List[str]): List of text chunks.
        
        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        return [self.client.embeddings.create(model=self.model, input=chunk) for chunk in text_chunks]

    def compute_breakpoints(self, similarities, method="percentile", threshold=90):
        """
        Computes chunking breakpoints based on similarity drops.
        
        Args:
            similarities (List[float]): List of similarity scores between sentences.
            method (str): 'percentile', 'standard_deviation', or 'interquartile'. 
            threshold (float): Threshold value (percentile for 'percentile', std devs for 'standard_deviation').
        
        Returns:
            List[int]: Indices where chunk splits should occur.
        """
        if method == "percentile":
            threshold_value = np.percentile(similarities, threshold)
        elif method == "standard_deviation":
            mean = np.mean(similarities)
            std_dev = np.std(similarities)
            threshold_value = mean - (threshold * std_dev)
        elif method == "interquartile":
            q1, q3 = np.percentile(similarities, [25, 75])
            threshold_value = q1 - 1.5 * (q3 - q1)
        else:
            raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")
        
        return [i for i, sim in enumerate(similarities) if sim < threshold_value]

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text from the PDF.
        """
        mypdf = fitz.open(pdf_path)
        all_text = ""  # Initialize an empty string to store the extracted text
        for page in mypdf:
            all_text += page.get_text("text") + " "
        return all_text.strip()

    def cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.
        
        Returns:
            float: Cosine similarity.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_embedding(self, text, model="BAAI/bge-en-icl"):
        """
        Creates an embedding for the given text using OpenAI.
        
        Args:
            text (str): Input text.
            model (str): Embedding model name.
        
        Returns:
            np.ndarray: The embedding vector.
        """
        response = self.client.embeddings.create(model=model, input=text)
        return np.array(response.data[0].embedding)

    def generate_response(self, system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Generates a response from the AI model based on the system prompt and user message.
        
        Args:
            system_prompt (str): The system prompt to guide the AI's behavior.
            user_message (str): The user's message or query.
            model (str): The model to be used for generating the response. Default is "meta-llama/Llama-2-7B-chat-hf".
        
        Returns:
            dict: The response from the AI model.
        """
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response

    def evaluate_ai_response(self, query, ai_response, true_response, evaluate_system_prompt):
        """
        Evaluates the AI response by comparing it with the expected answer and assigning a score.
        
        Args:
            query (str): Search query.
            ai_response (dict): AI response.
            true_response (dict): True response.
            evaluate_system_prompt (str): Evaluation system prompt.
        
        Returns:
            dict: The evaluation response from the evaluation system prompt and evaluation prompt.
        """
        user_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {true_response['ideal_answer']}\n{evaluate_system_prompt}"
        evaluation_response = self.generate_response(evaluate_system_prompt, user_prompt)
        return evaluation_response

# Define the system prompt for the AI assistant
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

# Create the AI assistant
ai_assistant = Chunker()

# Define the user query
query = "What is the capital of France?"

# Extract text from a PDF file
pdf_path = "data/AI_Information.pdf"
extracted_text = ai_assistant.extract_text_from_pdf(pdf_path)

# Split text into sentences
sentences = extracted_text.split(". ")

# Generate embeddings for each sentence
embeddings = [ai_assistant.get_embedding(sentence) for sentence in sentences]

# Compute similarity differences
similarities = [ai_assistant.cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

# Compute breakpoints using the percentile method with a threshold of 90
breakpoints = ai_assistant.compute_breakpoints(similarities, method="percentile", threshold=90)

# Split text into semantic chunks
text_chunks = ai_assistant.split_into_chunks(sentences, breakpoints)

# Create chunk embeddings using the create_embeddings function
chunk_embeddings = ai_assistant.create_embeddings(text_chunks)

# Create chunk embeddings using the create_embeddings function
chunk_embeddings = ai_assistant.create_embeddings(text_chunks)

# Create the user prompt based on the top chunks
user_prompt = "n".join([f"Context {i + 1}:n{chunk}n=====================================n" for i, chunk in enumerate(top_chunks)])

# Generate AI response
