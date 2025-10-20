import os

from jet.code.markdown_utils._preprocessors import remove_markdown_links
from jet.file.utils import save_file
from jet.logger import logger

text_with_sample_md_links = """
Visit [Google](https://www.google.com) now
[ ](/)
[ ](/db/ \"Database\") [ ](/threads/ \"Threads\")
With [Go to home](/) text
[ Sample 1 ](/db/ \"Database\") [ Sample 2 ](/threads/ \"Threads\")

[ ](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F%2F&src=sdkpreparse) [ ](https://twitter.com/intent/tweet?text=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F) [ ](https://web.whatsapp.com/send?text=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F)

[ ](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F&title=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&source=gamerant.com&summary=Isekai%20anime%20is%20inescapable%2C%20with%20each%20season%20containing%20a%20couple%20of%20shows.%20Here%20are%20the%202025%20isekai%20anime%20series%20announced%20so%20far.) [ ](https://bsky.app/intent/compose?text=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F) [ ](https://www.reddit.com/submit?url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F) [ ](http://share.flipboard.com/bookmarklet/popout?v=2&title=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F&utm_campaign=tools&utm_medium=article-share&utm_source=gamerant.com)

[ ](mailto:?Subject=Every New Isekai Anime Announced For 2025 \(So Far\)&Body=Check%20this%20out%21%0Ahttps%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F)

Facebook X LinkedIn Reddit Flipboard Copy link [ Email ](mailto:?Subject=Every New Isekai Anime Announced For 2025 \(So Far\)&Body=Check%20this%20out%21%0Ahttps://gamerant.com/new-isekai-anime-2025/)
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    result = remove_markdown_links(text_with_sample_md_links, remove_text=True)

    logger.debug(f'\n"""\n{text_with_sample_md_links}\n"""')
    logger.success(f'\n"""\n{result}\n"""')

    save_file(result, f"{output_dir}/result.txt")
