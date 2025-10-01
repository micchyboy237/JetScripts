from typing import Dict, List, Optional, TypedDict, Union
from dataclasses import dataclass
import re

class DocumentChunk(TypedDict):
    content: str
    metadata: Dict[str, Union[str, int]]
    chunk_id: str

class PreprocessingConfig(TypedDict, total=False):
    chunk_size: int
    chunk_overlap: int
    add_section_summaries: bool
    context_window_tokens: int
    strip_markdown: bool

@dataclass
class ProcessedDocument:
    original_content: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, str]
    preprocessing_config: PreprocessingConfig

class DocumentPreprocessor:
    """
    Preprocesses documents for RAG systems with support for plain text and markdown.
    Implements contextual retrieval best practices.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        # Default configuration
        self.default_config: PreprocessingConfig = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'add_section_summaries': True,
            'context_window_tokens': 50,
            'strip_markdown': False
        }
        
        # Merge provided config with defaults
        if config:
            self.config = {**self.default_config, **config}
        else:
            self.config = self.default_config.copy()
    
    def preprocess_document(self, content: str, metadata: Optional[Dict[str, str]] = None) -> ProcessedDocument:
        """
        Main preprocessing pipeline for RAG documents.
        
        Args:
            content: Raw document content (plain text or markdown)
            metadata: Optional document metadata
            
        Returns:
            ProcessedDocument with chunks and metadata
        """
        if metadata is None:
            metadata = {}
        
        # Clean and normalize content
        cleaned_content = self._clean_content(content)
        
        # Enhance with section summaries if needed
        if self.config.get('add_section_summaries', True):
            enhanced_content = self._add_section_summaries(cleaned_content)
        else:
            enhanced_content = cleaned_content
        
        # Split into chunks with context
        chunks = self._split_with_context(enhanced_content, metadata)
        
        return ProcessedDocument(
            original_content=content,
            chunks=chunks,
            metadata=metadata,
            preprocessing_config=self.config
        )
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', content.strip())
        
        # Optionally strip markdown but keep semantic structure
        if self.config.get('strip_markdown', False):
            cleaned = self._strip_markdown(cleaned)
            
        return cleaned
    
    def _strip_markdown(self, content: str) -> str:
        """Strip markdown syntax while preserving text content."""
        # Remove headers but keep the text
        content = re.sub(r'#+\s*', '', content)
        # Remove bold/italic but keep text
        content = re.sub(r'[\*_]{1,2}(.*?)[\*_]{1,2}', r'\1', content)
        # Remove links but keep anchor text
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
        # Remove code blocks but keep content
        content = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', content)
        # Remove blockquotes
        content = re.sub(r'^\s*>+\s*', '', content, flags=re.MULTILINE)
        # Remove horizontal rules
        content = re.sub(r'^\s*[*\-_]{3,}\s*$', '', content, flags=re.MULTILINE)
        
        return content
    
    def _add_section_summaries(self, content: str) -> str:
        """Add brief summaries to document sections."""
        lines = content.split('\n')
        enhanced_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            enhanced_lines.append(line)
            
            # Detect section headers (simplified)
            is_section_header = (
                line.strip() and 
                (line.startswith('#') or 
                 i == 0 or 
                 (line.endswith(':') and len(line) < 100) or
                 (line.isupper() and len(line) < 100))  # All caps headers
            )
            
            if is_section_header:
                # Get next few lines for context
                next_lines = []
                j = i + 1
                while j < len(lines) and j < i + 10 and not lines[j].strip().startswith('#'):
                    if lines[j].strip():
                        next_lines.append(lines[j])
                    j += 1
                
                summary = self._generate_section_summary(line, next_lines)
                if summary:
                    enhanced_lines.append(f"*Summary: {summary}*")
                    enhanced_lines.append("")  # Add spacing
            
            i += 1
        
        return '\n'.join(enhanced_lines)
    
    def _generate_section_summary(self, header: str, next_lines: List[str]) -> Optional[str]:
        """Generate a brief summary for a section."""
        if not next_lines:
            return None
            
        sample_text = ' '.join(next_lines[:3])  # Use first few lines
        if len(sample_text) > 50:
            # Simple extraction - in practice, use more sophisticated methods
            sentences = re.split(r'[.!?]', sample_text)
            if sentences and len(sentences[0]) > 10:
                return sentences[0][:100] + "..."
        return sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
    
    def _split_with_context(self, content: str, metadata: Dict[str, str]) -> List[DocumentChunk]:
        """
        Split document into chunks with contextual information prepended.
        Implements Contextual Retrieval pattern.
        """
        if not content.strip():
            return []
            
        chunks = []
        words = content.split()
        chunk_size = self.config.get('chunk_size', 1000)
        chunk_overlap = self.config.get('chunk_overlap', 200)
        
        # Ensure we don't have negative or zero chunk size
        chunk_size = max(100, chunk_size)
        chunk_overlap = min(chunk_overlap, chunk_size - 50)  # Ensure reasonable overlap
        
        i = 0
        chunk_index = 0
        while i < len(words):
            end_index = min(i + chunk_size, len(words))
            chunk_words = words[i:end_index]
            chunk_content = ' '.join(chunk_words)
            
            # Add contextual information to chunk
            contextualized_content = self._add_chunk_context(chunk_content, metadata, i, chunk_index)
            
            chunk = DocumentChunk(
                content=contextualized_content,
                metadata={
                    'word_count': len(chunk_words),
                    'start_position': i,
                    'end_position': end_index,
                    'source': metadata.get('source', 'unknown'),
                    'chunk_index': chunk_index,
                    **{k: v for k, v in metadata.items() if k != 'source'}  # Avoid duplicate source
                },
                chunk_id=f"chunk_{chunk_index:04d}_{i:06d}"
            )
            chunks.append(chunk)
            
            # Move to next chunk position
            if end_index == len(words):
                break
                
            i += chunk_size - chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def _add_chunk_context(self, chunk_content: str, metadata: Dict[str, str], position: int, chunk_index: int) -> str:
        """Prepend contextual information to chunk (Contextual Retrieval pattern)."""
        context_parts = []
        
        # Add document-level context
        if 'title' in metadata:
            context_parts.append(f"This section is from '{metadata['title']}'")
        if 'document_type' in metadata:
            context_parts.append(f"Document type: {metadata['document_type']}")
        
        # Add chunk-specific context
        context_parts.append(f"This chunk (part {chunk_index + 1}) discusses: {self._extract_chunk_topic(chunk_content)}")
        
        context = ". ".join(context_parts)
        return f"Context: {context}. Original content: {chunk_content}"

    def _extract_chunk_topic(self, chunk_content: str) -> str:
        """Extract a brief topic description for contextual information."""
        # Clean the content for topic extraction
        clean_content = re.sub(r'Context:.*?Original content:', '', chunk_content)
        sentences = re.split(r'[.!?]', clean_content)
        
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                return first_sentence[:80] + "..."
        
        # Fallback: use first 50 characters
        words = clean_content.split()[:10]
        if words:
            return ' '.join(words) + "..."
        
        return "key concepts from this document section"

    def preprocess_query(self, query: str, conversation_history: Optional[List[str]] = None) -> str:
        """
        Preprocess user queries with conversation context.
        
        Args:
            query: Current user query
            conversation_history: Previous queries and responses for context
            
        Returns:
            Contextualized query for improved retrieval
        """
        if not conversation_history:
            return query
        
        # Add recent conversation context to query
        recent_history = ' '.join(conversation_history[-2:])  # Last 2 exchanges
        contextualized_query = f"Previous context: {recent_history}. Current question: {query}"
        
        return contextualized_query


if __name__ == "__main__":
    import os

    from jet.file.utils import save_file

    md_sample = """
Sample title

# Project Overview
Welcome to our **project**! This is an `introduction` to our work, featuring a [website](https://project.com).

![Project Logo](https://project.com/logo.png)

> **Note**: Always check the [docs](https://docs.project.com) for updates.

## Features
- [ ] Task 1: Implement login
- [x] Task 2: Add dashboard
- Task 3: Optimize performance

### Technical Details
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

#### API Endpoints
| Endpoint       | Method | Description           |
|----------------|--------|-----------------------|
| /api/users     | GET    | Fetch all users       |
| /api/users/{id}| POST   | Create a new user     |

##### Inline Code
Use `print("Hello")` for quick debugging.

###### Emphasis
*Italic*, **bold**, and ***bold italic*** text are supported.

<div class="alert">This is an HTML block.</div>
<span class="badge">New</span> inline HTML.

[^1]: This is a footnote reference.
[^1]: Footnote definition here.

## Unordered list
- List item 1
    - Nested item
- List item 2
- List item 3

## Ordered list
1. Ordered item 1
2. Ordered item 2
3. Ordered item 3

## Inline HTML
<span class="badge">New</span> inline HTML
"""

    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    config = {
        "chunk_size": 100,
        "chunk_overlap": 0,
        "strip_markdown": True,
        "add_section_summaries": True,
    }
    size_processor = DocumentPreprocessor(config)
        
    result = size_processor.preprocess_document(md_sample)
    save_file(result, f"{output_dir}/result.json")
