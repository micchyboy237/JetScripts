import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    dimensions = None
    # dimensions = 512
    # model_name: EmbedModelType = "mxbai-embed-large"
    # model_name: EmbedModelType = "nomic-embed-text"
    # model_name: EmbedModelType = "all-MiniLM-L6-v2"
    model_name: EmbedModelType = "embeddinggemma"
    # model_name: EmbedModelType = "static-retrieval-mrl-en-v1"
    # Same example queries
    queries = [
        "Top 10 isekai anime 2025 with release date, synopsis, number of episode, airing status",
    ]

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)

    # Same sample documents
    sample_docs = [
        "[Mushoku Tensei Jobless reincarnation season 3 Rudeus](https://static0.srcdn.com/wordpress/wp-content/uploads/2025/09/mushoku-tensei-jobless-reincarnation-season-3-rudeus.jpg?q=70&fit=crop&w=825&dpr=1)\n\nMushoku Tensei Jobless reincarnation season 3 Rudeus\n\n*Mushoku Tensei: Jobless Reincarnation* has become one of the most popular isekai, and deservedly so.",
        "While *Mushoku Tensei: Jobless Reincarnation*'s protagonist has sparked criticism, the anime features one of the most fascinating fantasy worlds and character development of a flawed man.",
        "Luckily, *[Mushoku Tensei: Jobless Reincarnation Season 3](https://screenrant.com/mushoku-tensei-jobless-reincarnation-season-3-anime-announced/)* will release soon featuring the Turning Point 4 arc in 2026.",
        "The recent trailer for *Mushoku Tensei: Jobless Reincarnation Season 3*, released at Anime Expo 2025, teases Eris's return and her meeting with Rudeus.",
        "Additionally, recent reports from Toho point to a spring 2026 premiere, so fans won't have to wait as much to see how the relationship between Rudeus, Sylphiette, and Roxy develops.",
        "!",
        "[\n\n](https://static0.srcdn.com/wordpress/wp-content/uploads/2024/04/kq5zfgpirwc-hd.jpg?q=49&fit=crop&w=450&h=225)](https://video.srcdn.com/2024/04/2-op-2-ii-2-1713895387.mp4)\n\n\n\n### Cast\n\n[See All](https://screenrant.com/db/tv-show/mushoku-tensei-jobless-reincarnation/#cast-cand-crew)\n\n* !",
        "[Cast Placeholder Image](https://static0.srcdn.com/wordpress/wp-content/uploads/sharedimages/2024/07/screen-hub-cast-placeholder-1.png?q=49&fit=crop&w=50&h=65&dpr=2)\n\n  Yumi Uchiyama\n\n  Rudeus Greyrat's Former Self(voice)\n* !",
        "[Cast Placeholder Image](https://static0.srcdn.com/wordpress/wp-content/uploads/sharedimages/2024/07/screen-hub-cast-placeholder-1.png?q=49&fit=crop&w=50&h=65&dpr=2)\n\n  Tomokazu Sugita\n\n  Pilemon Notos Greyrat (voice)\n\n \n\nMushoku Tensei: Jobless Reincarnation follows a man who, after being reincarnated in a magical world, sets out to live his life without regrets.",
        "Adopting the name Rudeus Greyrat, he embarks on a journey of personal growth and adventure, armed with both his past life's knowledge and newfound magical abilities.",
        "The series explores themes of redemption, self-discovery, and the quest to overcome personal failures.",
        "**Main Genre**\n:   [Anime](/tag/anime/)\n\n**Creator(s)**\n:   Rifujin Na Magonote\n\n**Producers**\n:   Takahiro Yamanaka, Tomoyuki Ohwada, Sho Osuga, Takurou Hatakeyama, Ryousuke Imai, Mitsuteru Hishiyama, Nobuhiro Oosawa, Fuminori Yamazaki, Sou Yurugi, Takumi Morii\n\n**Seasons**\n:   2\n\n[Powered by\n\n###### ScreenRant logo](https://screenrant.com/db/)\n\nExpand\nCollapse\n\n* [!",
        "[Anime](https://static0.srcdn.com/wordpress/wp-content/uploads/2024/01/chihiro-sad-in-spirited-away-1.jpg?q=50&fit=crop&w=32&h=32&dpr=1.5)\n\n  Anime](/anime/ \"Anime\")\n* [!",
        "[Mushoku Tensei: Jobless Reincarnation](https://static0.srcdn.com/wordpress/wp-content/uploads/2024/04/mushoku-tensei-jobless-reincarnation.jpg?q=50&fit=crop&w=32&h=32&dpr=1.5)\n\n  Mushoku Tensei: Jobless Reincarnation](/db/tv-show/mushoku-tensei-jobless-reincarnation/ \"Mushoku Tensei: Jobless Reincarnation\")\n* [!",
        "[Reddit](https://www.reddit.com/submit?url=https%3A%2F%2Fscreenrant.com%2Fmost-anticipated-isekai-anime%2F)\n[Flipboard](http://share.flipboard.com/bookmarklet/popout?v=2&title=10%20Upcoming%20Isekai%20Anime%20We%20Guarantee%20Fans%20Will%20Obsess%20Over&url=https%3A%2F%2Fscreenrant.com%2Fmost-anticipated-isekai-anime%2F&utm_campaign=tools&utm_medium=article-share&utm_source=screenrant.com)\n[Copy link](javascript:;)\n[Email](mailto:?Subject=10 Upcoming Isekai Anime We Guarantee Fans Will Obsess Over&Body=Check%20this%20out%21%0Ahttps://screenrant.com/most-anticipated-isekai-anime/)\n\nClose\n\nThread\n\nSign in to your ScreenRant account\n\nWe want to hear from you!",
        "Share your opinions in the thread below and remember to keep it respectful.",
        "Be the first to post\n\nImages\n\nAttachment(s)\n\nPlease respect our [community guidelines](https://www.valnetinc.com/en/terms-of-use#community-guidelines).",
        "No links, inappropriate language, or spam.",
        "Your comment has not been saved\n\n[Send confirmation email](javascript:void(0))\n\nThis thread is open for discussion.",
        "Be the first to post your thoughts.",
        "* [Terms](https://www.valnetinc.com/en/terms-of-use)\n* [Privacy](/page/our-privacy-policy/)\n* [Feedback](/contact)\n\nRecommended\n\n[!",
        "[Netflix featured image - collage of Netflix tv series and movies behind the logo](https://static0.srcdn.com/wordpress/wp-content/uploads/2025/09/netflix-featured-image-collage-of-netflix-tv-series-and-movies-behind-the-logo.jpeg?q=49&fit=crop&w=266&h=350&dpr=2 \"Netflix's Movie Based On 48-Year-Old Children's Classic Has Cast Its Lead\")](/miss-nelson-is-missing-netflix-melissa-mccarthy-cast/)\n\n7 days ago\n\n### [Netflix's Movie Based On 48-Year-Old Children's Classic Has Cast Its Lead](/miss-nelson-is-missing-netflix-melissa-mccarthy-cast/ \"Netflix's Movie Based On 48-Year-Old Children's Classic Has Cast Its Lead\")\n\n[!",
        "[Superman and Lex Luthor squaring off in Superman (2025)](https://static0.srcdn.com/wordpress/wp-content/uploads/2025/09/superman-lex-luthor.jpg?q=49&fit=crop&w=266&h=350&dpr=2 \"James Gunn All But Confirms Man Of Tomorrow Villain With DCU Script Tease\")](/man-of-tomorrow-james-gunn-dcu-script-image-brainiac/)\n\n7 days ago\n\n### [James Gunn All But Confirms Man Of Tomorrow Villain With DCU Script Tease](/man-of-tomorrow-james-gunn-dcu-script-image-brainiac/ \"James Gunn All But Confirms Man Of Tomorrow Villain With DCU Script Tease\")\n\n[!",
        "[Emma-Watson](https://static0.srcdn.com/wordpress/wp-content/uploads/2025/09/emma-watson.jpg?q=49&fit=crop&w=266&h=350&dpr=2 \"Emma Watson Explains 7-Year Acting Hiatus\")](/emma-watson-acting-hiatus-explained/)\n\n6 days ago\n\n### [Emma Watson Explains 7-Year Acting Hiatus](/emma-watson-acting-hiatus-explained/ \"Emma Watson Explains 7-Year Acting Hiatus\")\n\n[!",
        "[AT-AT Walkers march forward on Hoth shooting blasts in The Empire Strikes Back](https://static0.srcdn.com/wordpress/wp-content/uploads/2023/01/battle-of-hoth-the-empire-strikes-back.jpg?q=49&fit=crop&w=266&h=350&dpr=2 \"Star Wars Built A Life-Size AT-AT For The Mandalorian and Grogu\")](/star-wars-the-mandalorian-and-grogu-life-size-at-at/)\n\n6 days ago\n\n### [Star Wars Built A Life-Size AT-AT For The Mandalorian and Grogu](/star-wars-the-mandalorian-and-grogu-life-size-at-at/ \"Star Wars Built A Life-Size AT-AT For The Mandalorian and Grogu\")\n\nMore from our brands\n\n!",
        "[CBR logo](./public/build/images/cbr-crossbrands-logo-black.svg?v=3.0 \"CBR\")\n\n[### The 40 Best Isekai Of All Time, Ranked](https://www.cbr.com/best-isekai-anime-ranked/)\n\n!",
        "[GameRant logo](./public/build/images/gr-crossbrands-logo-black.svg?v=3.0 \"GameRant\")\n\n[### Every New Isekai Anime Announced For 2024](https://gamerant.com/new-isekai-anime-2024-upcoming/)\n\n!",
        "[CBR logo](./public/build/images/cbr-crossbrands-logo-black.svg?v=3.0 \"CBR\")\n\n[### The 40 Best Isekai Anime Streaming on Crunchyroll](https://www.cbr.com/best-isekai-on-crunchyroll/)\n\n!",
        "[CBR logo](./public/build/images/cbr-crossbrands-logo-black.svg?v=3.0 \"CBR\")\n\n[### The Best Isekai Anime on Streaming (March 2025)](https://www.cbr.com/best-isekai-anime-streaming-now/)\n\n[!",
        "[new 2025 isekai anime](https://static0.gamerantimages.com/wordpress/wp-content/uploads/2025/03/new-2025-isekai-anime.jpg?q=50&fit=crop&w=680&h=511&dpr=1.5 \"Every New Isekai Anime Announced For 2025 (So Far)\")](https://gamerant.com/new-isekai-anime-2025/)\n\n!",
        "[GameRant logo](./public/build/images/gr-crossbrands-logo-black.svg?v=3.0 \"GameRant\")\n\n### [Every New Isekai Anime Announced For 2025 (So Far)](https://gamerant.com/new-isekai-anime-2025/ \"Every New Isekai Anime Announced For 2025 (So Far)\")\n\n[!",
        "[The-Greatest-Isekai-Anime-Of-All-Time-(November-2024)-B](https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/12/the-greatest-isekai-anime-of-all-time-november-2024-b.jpg?q=50&fit=crop&w=383&h=247&dpr=1.5 \"The Greatest Isekai Anime Of All Time (April 2025)\")](https://gamerant.com/best-isekai-anime-to-watch-ranked/)\n\n!"
    ]
    search_engine.add_documents(sample_docs)

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")
