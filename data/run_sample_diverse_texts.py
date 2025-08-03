
import os
from typing import List
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

texts = [
    "Destination A: Pristine beaches with white sand, ancient temples, vibrant festivals, and local cuisine.",
    "Destination B: Sandy beaches, historic ruins, cultural museums, and traditional dance performances.",
    "Destination C: Tropical beachfront, coral reefs for snorkeling, modern art galleries, and luxury resorts.",
    "Destination D: Golden beaches, UNESCO heritage sites, local artisan markets, and coastal hiking trails.",
    "Destination E: Rugged mountain trails, alpine villages, folk music festivals, and scenic cable cars.",
    "Destination F: Urban city with cultural landmarks, historic theaters, street food markets, and river cruises.",
    "Destination G: Secluded beaches, minimal cultural sites, excellent diving spots, and eco-friendly lodges."
]

if __name__ == "__main__":
    diverse_result_texts: List[str] = sample_diverse_texts(texts)

    save_file(diverse_result_texts, f"{OUTPUT_DIR}/diverse_result_texts.json")
