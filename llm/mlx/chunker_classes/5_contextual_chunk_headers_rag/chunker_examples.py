import fitz
import numpy as np
import json
from openai import OpenAI
from tqdm import tqdm

class Chunker:
    def __init__(self, pdf_path, model="meta-llama/Llama-3.2-3B-Instruct"):
        self.pdf_path = pdf_path
        self.model = model
        self.openai = OpenAI()
        self.client = self.openai.chat

    def extract_text_from_pdf(self):
        mypdf = fitz.open(self.pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            text = page.get_text("text")
            all_text += text
        return all_text

    def chunk_text_with_headers(self, text, chunk_size=1000, overlap=200):
        text_chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if i + overlap < len(text):
                chunk += text[i+overlap:i+chunk_size+overlap]
            text_chunks.append({
                'text': chunk,
                'header': self.generate_chunk_header(chunk)
            })
        return text_chunks

    def generate_chunk_header(self, chunk, model="meta-llama/Llama-3.2-3B-Instruct"):
        system_prompt = "Generate a concise and informative title for the given text."
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ]
        )
        return response.choices[0].message.content.strip()

    def create_embeddings(self, text, model="BAAI/bge-en-icl"):
        response = self.openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    def semantic_search(self, query, embeddings, k=2):
        similarities = []
        for i, embedding in enumerate(embeddings):
            similarity = self.cosine_similarity(embedding, self.create_embeddings(query))
            similarities.append((i, similarity))
        top_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return [embeddings[i] for i, _ in top_chunks]

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def generate_response(self, system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response

    def evaluate_response(self, system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
        evaluation_prompt = f"""
User Query: {user_message}
AI Response: {self.generate_response(system_prompt, user_message, model)}
True Answer: {true_answer}
{system_prompt}
"""
        evaluation_response = self.generate_response(system_prompt, evaluation_prompt)
        return evaluation_response.choices[0].message.content

def main():
    pdf_path = "path_to_your_pdf_file.pdf"
    model = "meta-llama/Llama-3.2-3B-Instruct"
    chunker = Chunker(pdf_path, model)
    
    # Extract text from PDF
    text = chunker.extract_text_from_pdf()
    print(f"Extracted text from PDF: {text}")
    
    # Chunk text into chunks of size 1000 with overlap of 200
    text_chunks = chunker.chunk_text_with_headers(text, chunk_size=1000, overlap=200)
    print(f"Chunked text into chunks: {text_chunks}")
    
    # Generate chunk headers
    chunk_headers = chunker.generate_chunk_header(text)
    print(f"Generated chunk headers: {chunk_headers}")
    
    # Create embeddings for a given text
    embeddings = chunker.create_embeddings(text, model)
    print(f"Created embeddings for text: {embeddings}")
    
    # Semantic search for a given query
    query = "example query"
    embeddings = chunker.create_embeddings(text, model)
    top_chunks = chunker.semantic_search(query, embeddings, k=2)
    print(f"Top chunks for query: {top_chunks}")
    
    # Evaluate the generated response
    system_prompt = "example system prompt"
    user_message = "example user message"
    model = "meta-llama/Llama-3.2-3B-Instruct"
    evaluation_response = chunker.evaluate_response(system_prompt, user_message, model)
    print(f"Evaluated response: {evaluation_response}")

if __name__ == "__main__":
    main()