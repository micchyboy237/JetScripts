from jet.adapters.turftopic.embed import LlamaCppEncoder
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing encoder and vectorizer...")
encoder = LlamaCppEncoder()
vectorizer = CountVectorizer(min_df=1, stop_words="english")

# --- 1. Diverse Corpus Construction ---
corpus = [
    # --- Cluster A: Climate & Environment (Strong Signal) ---
    "Global warming is causing ice caps to melt at an unprecedented rate.",
    "Renewable energy sources like solar and wind are essential for a sustainable future.",
    "Carbon emissions must be reduced to meet the Paris Agreement targets.",
    "Deforestation in the Amazon is contributing significantly to climate change.",
    "Electric vehicles are becoming a viable alternative to fossil fuel cars.",
    # --- Cluster B: Technology & AI (Strong Signal) ---
    "Large language models are transforming the field of natural language processing.",
    "Quantum computing promises to solve complex problems faster than classical computers.",
    "Cybersecurity threats are evolving with the rise of cloud infrastructure.",
    "Machine learning algorithms are optimizing supply chain logistics globally.",
    # --- Cluster C: Sports (Weak Signal - only 2 docs) ---
    "The World Cup final attracted billions of viewers worldwide.",
    "Basketball players require intense training and strategic teamwork.",
    # --- Outliers & Noise (The "Junk" Drawer) ---
    "The quick brown fox jumps over the lazy dog.",  # Classic pangram, no topic
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",  # Placeholder text
    "My cat loves to sleep on the keyboard while I work.",  # Personal/Irrelevant
    "404 Error: Page not found. Please try again later.",  # Technical noise
    "Buy one get one free on all shoes this weekend!",  # Commercial spam
    # --- Ambiguous/Hybrid Documents ---
    "Green technology investments are rising in the stock market.",  # Mix of Finance/Tech/Climate
    "Data centers consume massive amounts of electricity, raising environmental concerns.",  # Mix of Tech/Climate
]

print(f"Corpus Size: {len(corpus)} documents")
print("-" * 80)

# --- 2. Strategy 1: The "Unrelated Seed" Mistake ---
print("\n🔴 ATTEMPT 1: Using Unrelated Words in One Seed")
print("Seed: 'climate change basketball'")
try:
    model_bad = KeyNMF(
        3,
        seed_phrase="climate change basketball",
        top_n=5,
        random_state=42,
        vectorizer=vectorizer,
        encoder=encoder,
    )
    model_bad.fit(corpus)
    model_bad.print_topics()
    print(
        "Observation: Topics are likely diluted or incoherent because the vector averages two distant concepts."
    )
except Exception as e:
    print(f"Error: {e}")

# --- 3. Strategy 2: The "Broad Hypernym" Attempt ---
print("\n🟡 ATTEMPT 2: Using a Broad Abstract Seed")
print("Seed: 'global trends and changes'")
try:
    model_broad = KeyNMF(
        3,
        seed_phrase="global trends and changes",
        top_n=5,
        random_state=42,
        vectorizer=vectorizer,
        encoder=encoder,
    )
    model_broad.fit(corpus)
    model_broad.print_topics()
    print(
        "Observation: This might pick up everything vaguely related to 'change', including noise."
    )
except Exception as e:
    print(f"Error: {e}")

# --- 4. Strategy 3: The "Targeted Union" Approach (Recommended) ---
print("\n🟢 ATTEMPT 3: Targeted Search for Specific Themes")
print("We will search for 'Climate' AND 'Tech' separately, ignoring Sports/Noise.")

# A. Find Climate Docs
model_climate = KeyNMF(
    2,
    seed_phrase="environment renewable energy carbon",
    top_n=5,
    random_state=42,
    vectorizer=vectorizer,
    encoder=encoder,
)
model_climate.fit(corpus)

# B. Find Tech Docs
model_tech = KeyNMF(
    2,
    seed_phrase="artificial intelligence computing software",
    top_n=5,
    random_state=42,
    vectorizer=vectorizer,
    encoder=encoder,
)
model_tech.fit(corpus)


# C. Analyze Results
def get_top_docs(model, corpus, threshold=0.1):
    """Returns docs with max topic score above threshold"""
    scores = model.document_topic_matrix.max(axis=1)
    return [(i, scores[i], corpus[i]) for i, s in enumerate(scores) if s > threshold]


print("\n--- 🌍 Climate-Related Documents Found ---")
climate_docs = get_top_docs(model_climate, corpus)
for idx, score, doc in sorted(climate_docs, key=lambda x: x[1], reverse=True):
    print(f"[Score: {score:.3f}] (Doc {idx}) {doc[:80]}...")

print("\n--- 💻 Tech-Related Documents Found ---")
tech_docs = get_top_docs(model_tech, corpus)
for idx, score, doc in sorted(tech_docs, key=lambda x: x[1], reverse=True):
    print(f"[Score: {score:.3f}] (Doc {idx}) {doc[:80]}...")

# --- 5. Handling Outliers ---
print("\n--- 🗑️ Outlier Analysis ---")
all_relevant_indices = set(
    [i for i, _, _ in climate_docs] + [i for i, _, _ in tech_docs]
)
outlier_indices = [i for i in range(len(corpus)) if i not in all_relevant_indices]

print(f"Documents identified as OUTLIERS/NOISE by our filters:")
for idx in outlier_indices:
    print(f'  - Doc {idx}: "{corpus[idx][:60]}..."')

print("\n" + "=" * 80)
print("SUMMARY:")
print("1. Unrelated seeds dilute results.")
print("2. Broad seeds capture noise.")
print("3. Multiple targeted seeds allow you to 'pick and choose' relevant topics")
print(
    "   while automatically filtering out outliers and unrelated clusters (like Sports)."
)
