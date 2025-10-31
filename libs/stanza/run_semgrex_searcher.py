# --------------------------------------------------------------
#  example_search_and_filter_documents.py (FIXED)
# --------------------------------------------------------------

from __future__ import annotations
from typing import List
from stanza.server import CoreNLPClient
from jet.libs.stanza.semgrex_searcher import SemgrexSearcher, SemgrexMatch

# === 1. Start CoreNLP client ===
client = CoreNLPClient(
    annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'],
    timeout=30000,
    memory='6G',
    be_quiet=True,
)

# === 2. Initialize searcher ===
searcher = SemgrexSearcher(client)

# === 3. Long documents ===
documents: List[str] = [
    """BRCA1 is a tumor-suppressor gene that plays a crucial role in DNA repair.
       The protein regulates cell-cycle checkpoints and maintains genomic stability.
       Mutations in BRCA1 are associated with increased risk of breast cancer.""",

    """Tesla announced a new battery technology at its annual shareholder meeting.
       CEO Elon Musk revealed that the 4680 cell will be produced in-house.
       The company plans to scale production to 100 GWh by 2025.""",

    """I love hiking in the mountains. The fresh air clears my mind."""
]

# === 4. NAMED PATTERN (critical!) ===
pattern = "{pos:/NN.*/}=subject >nsubj {pos:/VB.*/}=verb"

# === 5. Search all documents ===
all_matches: List[SemgrexMatch] = searcher.search_documents(documents, pattern)
print(f"Found {len(all_matches)} matches across {len(documents)} documents.\n")

# === 6. Safe pretty-printer ===
def pp_match(m: SemgrexMatch) -> str:
    try:
        subj = next(n for n in m["nodes"] if n["name"] == "subject")
        verb = next(n for n in m["nodes"] if n["name"] == "verb")
        return (f"[Doc {m['doc_index']}] Sent {m['sentence_index']} | "
                f"{subj['text']} {verb['text']} "
                f"(subj-tag={subj['attributes'].get('tag')}, "
                f"verb-tag={verb['attributes'].get('tag')})")
    except StopIteration:
        return f"[Doc {m['doc_index']}] Sent {m['sentence_index']} | <unnamed nodes>"

# === 7. FILTER: Scientific doc ===
sci_matches = searcher.filter_matches(all_matches, doc_index=0)
print("=== FILTER: Scientific document (doc_index == 0) ===")
for m in sci_matches:
    print(pp_match(m))
print()

# === 8. FILTER: Subject contains "BRCA" ===
brca_matches = searcher.filter_matches(all_matches, node_text_contains="BRCA")
print("=== FILTER: Subject contains 'BRCA' ===")
for m in brca_matches:
    print(pp_match(m))
print()

# === 9. FILTER: Past tense verbs in news (VBD) ===
news_past = searcher.filter_matches(all_matches, doc_index=1, node_attr={"tag": "VBD"})
print("=== FILTER: News, past tense (VBD) ===")
for m in news_past:
    print(pp_match(m))
print()

# === 10. Cleanup ===
client.stop()