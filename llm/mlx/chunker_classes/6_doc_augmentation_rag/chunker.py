```python
import json
import numpy as np
import re
import simplevectorstore
import pymupdf
from pymuPDF import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chunker:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def chunk_text(self, text, n, overlap):
        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i:i + n])
        return chunks

    def create_embeddings(self, text, model="BAAI/bge-en-icl"):
        response = self.model.generate(text, max_length=10, temperature=0)
        return response

    def process_document(self, pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=3):
        reader = PdfReader(pdf_path)
        text = reader.get_text()
        text_chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        vector_store = simplevectorstore.SimpleVectorStore()
        for i, chunk in enumerate(text_chunks):
            chunk_embedding_response = self.create_embeddings(chunk)
            chunk_embedding = chunk_embedding_response.data[0].embedding
            vector_store.add_item(
                text=chunk,
                embedding=chunk_embedding,
                metadata={"type": "chunk", "index": i}
            )
            questions = self.generate_questions(chunk, num_questions=questions_per_chunk)
            for j, question in enumerate(questions):
                question_embedding_response = self.create_embeddings(question)
                question_embedding = question_embedding_response.data[0].embedding
                vector_store.add_item(
                    text=question,
                    embedding=question_embedding,
                    metadata={"type": "question", "chunk_index": i, "original_chunk": chunk}
                )
        return text_chunks, vector_store

    def generate_questions(self, text_chunk, num_questions=5, model="meta-llama/Llama-3.2-3B-Instruct"):
        system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only the provided text. Focus on key information and concepts."
        user_prompt = f"""
Based on the following text, generate {num_questions} different questions that can be answered using only this text:
{text_chunk}
Format your response as a numbered list of questions only, with no additional text.
"""
        response = self.model.generate(user_prompt, max_length=10, temperature=0.7)
        questions_text = response.choices[0].message.content.strip()
        questions = []
        for line in questions_text.split('n'):
            cleaned_line = re.sub(r'^d+.s*', '', line.strip())
            if cleaned_line and cleaned_line.endswith('? '):
                questions.append(cleaned_line)
        return questions

    def extract_text_from_pdf(self, pdf_path):
        reader = pymupdf.PdfReader(pdf_path)
        text = reader.get_text()
        return text

    def semantic_search(self, query, vector_store, k=5):
        search_results = []
        for i in range(1, len(vector_store.texts) + 1):
            chunk = vector_store.texts[i - 1]["text"]
            similarity = vector_store.texts[i - 1]["similarity"]
            search_results.append({
                "metadata": vector_store.texts[i - 1]["metadata"],
                "text": chunk,
                "similarity": similarity
            })
        search_results = sorted