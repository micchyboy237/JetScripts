```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI

class Chunker:
    def __init__(self, model="BAAI/bge-en-icl"):
        self.model = model
        self.client = OpenAI()

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file and prints the first `num_chars` characters.
        
        Args:
            pdf_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text from the PDF.
        """
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            text = page.get_text("text")
            all_text += text
        return all_text

    def chunk_text(self, text, n, overlap):
        """
        Chunks the given text into segments of n characters with overlap.
        
        Args:
            text (str): The text to be chunked.
            n (int): The number of characters in each chunk.
            overlap (int): The number of overlapping characters between chunks.
        
        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i:i + n])
        return chunks

    def create_embeddings(self, text):
        """
        Creates embeddings for the given text using the specified OpenAI model.
        
        Args:
            text (str): The input text for which embeddings are to be created.
        
        Returns:
            dict: The response from the OpenAI API containing the embeddings.
        """
        response = self.client.embeddings.create(model=self.model, input=text)
        return response

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
        response = self.client.chat.completions.create(model=model, temperature=0, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
        return response

    def evaluate_response(self, query, ai_response, data):
        """
        Evaluates the AI response based on the user query, AI response, and true response.
        
        Args:
            query (str): The user query.
            ai_response (dict): The AI response.
            data (dict): The true response.
        
        Returns:
            dict: The evaluation response.
        """
        evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1.  If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."
        evaluation_prompt = f"User Query: {query}\nAI Response: {ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"
        evaluation_response = self.generate_response(evaluate_system_prompt, evaluation_prompt)
        return evaluation_response

    def semantic_search(self, query, text_chunks, response_data, k=2):
        """
        Performs semantic search to find the top k most relevant text chunks for the query.
        
        Args:
            query (str): The user query.
            text_chunks (List[str]): The list of text chunks.
            response_data (dict): The response data.
            k (int): The number of top chunks to return. Default is 2.
        
        Returns:
            List[str]: The top k most relevant text chunks.
        """
        top_chunks = []
        for chunk in text_chunks:
            similarity = self.cosine_similarity(response_data[0]['vector'], self.create_embeddings(chunk))
            top_chunks.append((chunk, similarity))
        top_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in top_chunks[:k]]

    def cosine_similarity(self, vec1, vec2):
        """
        Calculates the cosine similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): The first vector.
            vec2 (np.ndarray): The second vector.
        
        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```