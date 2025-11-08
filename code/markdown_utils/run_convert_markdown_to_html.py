import os
import shutil

from jet.file.utils import save_file
from jet.code.markdown_utils._converters import convert_markdown_to_html

md_content1 = """
| Anime | Japanese Title | Season | Studio | Based On |
| --- | --- | --- | --- | --- |
| Head Start at Birth Season 2 | 0-saiji Start Dash Monogatari Season 2 | Winter 2025 | N/A | Light Novel/Manga by Umika |
| Ishura 2nd Season | Ishura 2nd Season | Passione | Light Novel by Keiso |
| Re:ZERO -Starting Life in Another World- Season 3 | Re:Zero kara Hajimeru Isekai Seikatsu 3rd Season | White Fox | Light Novel by Tappei Nagatsuki |
| I've Been Killing Slimes for 300 Years and Maxed Out My Level Season 2 | Slime Taoshite 300-nen, Shiranai Uchi ni Level Max ni Nattemashita: Sono Ni | Spring 2025 | Teddy | Light Novel by Kisetsu Morita |
| KonoSuba: God's Blessing on This Wonderful World! 3 OVA | Kono Subarashii Sekai ni Shukufuku wo! 3: Bonus Stage | Drive | Light Novel by Natsume Akatsuki |
| The Rising of the Shield Hero Season 4 | Tate no Yuusha no Nariagari Season 4 | Summer 2025 | Kinema Citrus | Light Novel by Aneko Yusagi |
| Kakuriyo: Bed and Breakfast for Spirits Season 2 | Kakuriyo no Yadomeshi 2nd Season | Fall 2025 | Gonzo, Makaria | Light Novel by Midori Yuma |
| Campfire Cooking in Another World with My Absurd Skill Season 2 | Tondemo Skill de Isekai Hourou Meshi 2nd Season | MAPPA | Light Novel by Ren Eguchi |
| Isekai Quarter 3 | Isekai Quartet 3 | Studio PuYUKAI | N/A |
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    html1 = convert_markdown_to_html(md_content1)

    save_file(html1, f"{output_dir}/html1.html")
