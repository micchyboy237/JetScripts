import numpy as np
import ctranslate2

# Example: Batch of token IDs (shape: [batch_size, seq_len])
tokens_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # Shape: (2, 3)

# Convert to StorageView (zero-copy on CPU)
tokens_view = ctranslate2.StorageView.from_array(tokens_np)

# Verify
print(tokens_view.shape)  # (2, 3)
print(tokens_view.dtype)  # <class 'numpy.int32'>

# Pass to a model (e.g., Translator)
# translator = ctranslate2.Translator("model_path")
# results = translator.translate_batch([tokens_view])