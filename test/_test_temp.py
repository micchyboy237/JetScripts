import pytest
from _temp import DocumentPreprocessor, ProcessedDocument

class TestDocumentPreprocessor:
    @pytest.fixture
    def processor(self):
        return DocumentPreprocessor()
    
    @pytest.fixture
    def sample_markdown_content(self):
        return """
        # Document Title
        
        This is an important document about **machine learning**.
        
        ## Key Concepts
        
        Neural networks are powerful models for pattern recognition.
        
        [Learn more](http://example.com)
        """
    
    @pytest.fixture
    def sample_plain_text(self):
        return """
        Project Overview
        
        This project implements RAG systems. RAG stands for Retrieval-Augmented Generation.
        We use it to enhance LLM capabilities with external knowledge.
        
        Technical Details:
        1. Preprocessing is critical
        2. Chunking strategy matters
        3. Context improves retrieval
        """
    
    def test_preprocess_document_returns_correct_type(self, processor, sample_markdown_content):
        result = processor.preprocess_document(sample_markdown_content)
        assert isinstance(result, ProcessedDocument)
    
    def test_preprocess_document_creates_chunks(self, processor, sample_markdown_content):
        result = processor.preprocess_document(sample_markdown_content)
        assert len(result.chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in result.chunks)
    
    def test_chunk_has_required_structure(self, processor, sample_markdown_content):
        result = processor.preprocess_document(sample_markdown_content)
        first_chunk = result.chunks[0]
        
        assert 'content' in first_chunk
        assert 'metadata' in first_chunk
        assert 'chunk_id' in first_chunk
        assert isinstance(first_chunk['metadata'], dict)
    
    def test_chunk_content_includes_context(self, processor, sample_markdown_content):
        result = processor.preprocess_document(
            sample_markdown_content, 
            {'title': 'Test Document', 'source': 'test'}
        )
        first_chunk = result.chunks[0]
        
        assert 'Context:' in first_chunk['content']
        assert 'Original content:' in first_chunk['content']
    
    def test_markdown_stripping(self, sample_markdown_content):
        # Test with explicit markdown stripping config
        stripping_config = {'strip_markdown': True, 'chunk_size': 500}
        stripping_processor = DocumentPreprocessor(stripping_config)
        
        result = stripping_processor.preprocess_document(sample_markdown_content)
        first_chunk_content = result.chunks[0]['content']
        
        # Check that markdown syntax is removed
        assert '**' not in first_chunk_content
        assert '##' not in first_chunk_content
    
    def test_chunk_size_respected(self, sample_plain_text):
        # Test with specific chunk size config
        chunk_size = 100
        size_processor = DocumentPreprocessor({'chunk_size': chunk_size})
        
        result = size_processor.preprocess_document(sample_plain_text)
        for chunk in result.chunks:
            # Approximate word count check (allow for context overhead)
            content_without_context = chunk['content'].split('Original content:')[-1]
            words = content_without_context.split()
            assert len(words) <= chunk_size + 20  # Allow some flexibility for word boundaries
    
    def test_preprocess_query_without_history(self, processor):
        query = "What is RAG?"
        result = processor.preprocess_query(query)
        assert result == query
    
    def test_preprocess_query_with_history(self, processor):
        query = "How does it work?"
        history = ["What is RAG?", "RAG is Retrieval-Augmented Generation..."]
        
        result = processor.preprocess_query(query, history)
        assert "Previous context:" in result
        assert "Current question:" in result
        assert query in result
    
    def test_metadata_preserved(self, processor, sample_plain_text):
        metadata = {'title': 'Test Doc', 'author': 'AI Engineer', 'version': '1.0'}
        result = processor.preprocess_document(sample_plain_text, metadata)
        
        assert result.metadata['title'] == 'Test Doc'
        assert result.metadata['author'] == 'AI Engineer'
        assert 'version' in result.chunks[0]['metadata']
    
    def test_empty_content(self, processor):
        result = processor.preprocess_document("")
        assert len(result.chunks) == 0
        assert result.original_content == ""
    
    def test_partial_config_merges_with_defaults(self):
        # Test that partial config properly merges with defaults
        partial_config = {'chunk_size': 500}
        processor = DocumentPreprocessor(partial_config)
        
        # Should have custom chunk_size but default for other values
        assert processor.config['chunk_size'] == 500
        assert processor.config['chunk_overlap'] == 200  # default
        assert processor.config['strip_markdown'] == False  # default
    
    def test_none_config_uses_all_defaults(self):
        processor = DocumentPreprocessor()
        assert processor.config['chunk_size'] == 1000
        assert processor.config['chunk_overlap'] == 200
        assert processor.config['add_section_summaries'] == True
        assert processor.config['strip_markdown'] == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])