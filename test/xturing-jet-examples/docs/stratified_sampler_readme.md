# Stratified Sampler

The `stratified_sampler` module provides tools for stratified sampling of text data, leveraging n-gram analysis and linguistic features to ensure representative sampling. It is designed to work with text datasets, offering functionality to filter, sort, and sample data based on various linguistic properties.

## Features Summary

- **Stratified Sampling**: Samples data while maintaining the distribution of linguistic features such as type-token ratio (TTR), sentence length, n-gram diversity, and starting n-grams.
- **N-Gram Analysis**: Extracts and analyzes n-grams to compute diversity and frequency, enabling sorting and filtering based on n-gram properties.
- **Flexible Input**: Supports both string lists and structured `ProcessedData` objects with source, target, and score attributes.
- **Quantile-Based Categorization**: Dynamically categorizes text based on quantiles of sentence length, TTR, n-gram diversity, and starting n-gram frequency.
- **Efficient Processing**: Uses `sklearn` for train-test splitting and includes progress tracking with `tqdm`.
- **Customizable Parameters**: Allows configuration of n-gram size, number of samples, and quantile limits for fine-tuned sampling.

## Usage Examples

### Example 1: Sampling Unique Strings

This example demonstrates how to use `StratifiedSampler` to sample unique strings from a dataset while preserving linguistic diversity.

```python
from stratified_sampler import StratifiedSampler

# Sample dataset
data = [
    "The quick brown fox jumps",
    "A fast red dog runs",
    "Slow green turtle walks"
]

# Initialize sampler with 50% sampling
sampler = StratifiedSampler(data, num_samples=0.5)

# Get unique strings
unique_strings = sampler.get_unique_strings()
print("Sampled Strings:", unique_strings)
```

**Expected Output**:

```
TTR Class Distribution: {'ttr_q1': 2, 'ttr_q2': 1}
Sentence Length Distribution: {'q1': 2, 'q2': 1}
N-Gram Diversity Distribution: {'ngram_q1': 3}
Starting N-Gram Distribution: {'start_q1': 3}
Sampled Strings: ['The quick brown fox jumps', 'A fast red dog runs']
```

### Example 2: Filtering Strings by N-Grams

This example shows how to filter and sort strings based on their starting n-grams.

```python
from stratified_sampler import StratifiedSampler

# Sample dataset
data = [
    "The quick brown fox",
    "The fast red dog",
    "A slow green turtle"
]

# Initialize sampler
sampler = StratifiedSampler(data, num_samples=2)

# Filter strings with 2-grams, taking top 2 per n-gram group
filtered_strings = sampler.filter_strings(n=2, top_n=2)
print("Filtered Strings:", filtered_strings)
```

**Expected Output**:

```
Filtered Strings: ['The quick brown fox', 'The fast red dog']
```

### Example 3: Sampling Structured Data

This example uses `ProcessedData` objects to sample structured data with source, target, and score attributes.

```python
from stratified_sampler import StratifiedSampler, ProcessedData

# Sample processed data
data = []
for source, target, categories, score in [
    ("The quick brown fox", "jumps", ["ttr_q1", "q1"], 0.9),
    ("A fast red dog runs", "barks", ["ttr_q2", "q2"], 0.8),
    ("Slow green turtle walks", "crawls", ["ttr_q1", "q1"], 0.7)
]:
    item = ProcessedData()
    item.source = source
    item.target = target
    item.category_values = categories
    item.score = score
    data.append(item)

# Initialize sampler
sampler = StratifiedSampler(data, num_samples=2)

# Get samples
samples = sampler.get_samples()
print("Sampled Data:", samples)
```

**Expected Output**:

```
Sampled Data: [
    {'source': 'The quick brown fox', 'target': 'jumps', 'score': 0.9},
    {'source': 'A fast red dog runs', 'target': 'barks', 'score': 0.8}
]
```

## Use Cases

1. **Dataset Preprocessing for NLP Models**:

   - Use `StratifiedSampler` to create balanced training and validation sets for natural language processing tasks, ensuring linguistic diversity in sampled data.
   - Example: Sampling dialogues for a chatbot while maintaining variety in sentence structures and vocabulary.

2. **Text Summarization and Filtering**:

   - Filter large text corpora to select representative sentences based on n-gram diversity and starting n-grams, useful for summarizing documents or extracting key phrases.
   - Example: Extracting diverse sentences from news articles for a summary.

3. **Linguistic Analysis**:

   - Analyze text data by categorizing sentences based on TTR, length, or n-gram properties, aiding in studies of linguistic patterns or text complexity.
   - Example: Studying language variation in social media posts by sampling tweets with diverse n-gram profiles.

4. **Data Augmentation**:
   - Sample subsets of text data for augmentation pipelines, ensuring that augmented datasets retain the original linguistic characteristics.
   - Example: Creating synthetic datasets for machine translation by sampling source-target pairs.

This module is particularly useful for researchers and developers working on text analysis, machine learning, or natural language processing tasks requiring representative sampling of text data.
