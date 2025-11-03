import os
from jet.file.utils import save_file
from span_marker import SpanMarkerModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the pre-trained model
model = SpanMarkerModel.from_pretrained(
    "lxyuan/span-marker-bert-base-multilingual-cased-multinerd",
).to("mps")

# Example text
text = """Title: Headhunted to Another World: From Salaryman to Big Four!
Isekai
Fantasy
Comedy
Release Date: January 1, 2025
Japanese Title: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi Studio

Studio: Geek Toys, CompTown

Based On: Manga

Creator: Benigashira

Streaming Service(s): Crunchyroll
Powered by
Expand Collapse
Plenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.
Related"""

# Predict entities
entities = model.predict(text)

print(entities)


output_dir = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
save_file(entities, f"{output_dir}/entities.json")
