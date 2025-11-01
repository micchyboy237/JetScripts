# import spacy
# from spacy import displacy
# from spacy.tokens import Span

# text = "Welcome to the Bank of China."

# nlp = spacy.blank("en")
# doc = nlp(text)

# doc.spans["sc"] = [
#     Span(doc, 3, 6, "ORG"),
#     Span(doc, 5, 6, "GPE"),
# ]

# displacy.serve(doc, style="span", port=5001)


import multiprocessing
import webbrowser
import spacy
from spacy import displacy
import signal
import sys
import time

text = (
    "When Sebastian Thrun started working on self-driving cars at Google in 2007, "
    "few people outside of the company took him seriously."
)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)


def run_server(style: str, port: int):
    """Run a displacy server for a given style and port."""
    displacy.serve(doc, style=style, port=port)


if __name__ == "__main__":
    # Start both servers in parallel
    ent_proc = multiprocessing.Process(target=run_server, args=("ent", 5001))
    dep_proc = multiprocessing.Process(target=run_server, args=("dep", 5002))

    ent_proc.start()
    dep_proc.start()

    # Give them a moment to start up
    time.sleep(1)
    webbrowser.open("http://localhost:5001")
    webbrowser.open("http://localhost:5002")

    def shutdown(*_):
        print("\nShutting down servers...")
        ent_proc.terminate()
        dep_proc.terminate()
        ent_proc.join()
        dep_proc.join()
        sys.exit(0)

    # Handle Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, shutdown)

    print("Servers running:")
    print("  ENT visualization → http://localhost:5001")
    print("  DEP visualization → http://localhost:5002")
    print("Press Ctrl+C to stop both.")

    # Keep the main process alive
    while True:
        time.sleep(1)

