from typing import List, Any, Union
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic


def load_sample_docs() -> List[str]:
    """Load sample documents from 20 Newsgroups for demo."""
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    return data.data[:100]  # Subset for quick demo


class CustomBERTopic(BERTopic):
    """Custom BERTopic subclass to adjust number of representative docs per topic."""
    
    def __init__(self, nr_repr_docs: int = 3, **kwargs: Any):
        super().__init__(**kwargs)
        self.nr_repr_docs = nr_repr_docs
    
    def _save_representative_docs(self, documents: pd.DataFrame) -> None:
        """Override to customize representative docs extraction."""
        repr_docs, _, _, _ = self._extract_representative_docs(
            self.c_tf_idf_,
            documents,
            self.topic_representations_,
            nr_samples=500,  # Adjustable: more for larger corpora
            nr_repr_docs=self.nr_repr_docs  # Custom: e.g., 5 instead of default 3
        )
        self.representative_docs_ = repr_docs


def fit_standard_model(docs: List[str]) -> BERTopic:
    """Fit standard BERTopic model (default 3 reps/topic)."""
    model = BERTopic(min_topic_size=10)  # Ensures topics have at least 10 docs
    model.fit(docs)
    return model


def fit_custom_model(docs: List[str], nr_repr_docs: int = 5) -> CustomBERTopic:
    """Fit custom model with adjustable reps/topic."""
    model = CustomBERTopic(min_topic_size=10, nr_repr_docs=nr_repr_docs)
    model.fit(docs)
    return model


def get_topic_reps_info(model: Union[BERTopic, CustomBERTopic]) -> pd.DataFrame:
    """Get topic info including Representative_Docs."""
    return model.get_topic_info()


# Demo usage
if __name__ == "__main__":
    docs = load_sample_docs()
    
    # Standard model
    standard_model = fit_standard_model(docs)
    standard_info = get_topic_reps_info(standard_model)
    print("Standard Model Representative Documents (Default 3 per topic):")
    for idx, row in standard_info.iterrows():
        print(f"\nTopic {row['Topic']}:")
        for i, doc in enumerate(row["Representative_Docs"], 1):
            print(f"  Doc {i}: {doc[:100]}...")  # Truncate for readability
    
    # Custom model
    custom_model = fit_custom_model(docs, nr_repr_docs=5)
    custom_info = get_topic_reps_info(custom_model)
    print("\nCustom Model Representative Documents (5 per topic):")
    for idx, row in custom_info.iterrows():
        print(f"\nTopic {row['Topic']}:")
        for i, doc in enumerate(row["Representative_Docs"], 1):
            print(f"  Doc {i}: {doc[:100]}...")  # Truncate for readability