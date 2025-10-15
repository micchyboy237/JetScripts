import os

from jet.utils.url_utils import remove_links
from jet.file.utils import save_file
from jet.logger import logger

text_with_sample_links = """
/
/db "Database" /threads/ "Threads" /sample-with-param?q=test /sample-with-fragment#test
https://thefilibusterblog.com/es/upcoming-isekai-anime-releases-for-2025-latest-announcements/
https://fyuu.net/new-isekai-anime-2025#content

https://www.facebook.com

https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F%2F&src=sdkpreparse  https://twitter.com/intent/tweet?text=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F

https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F&title=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&source=gamerant.com&summary=Isekai%20anime%20is%20inescapable%2C%20with%20each%20season%20containing%20a%20couple%20of%20shows.%20Here%20are%20the%202025%20isekai%20anime%20series%20announced%20so%20far. https://bsky.app/intent/compose?text=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F

Searxng link: http://jethros-macbook-air.local:3000/search?q=Top+10+isekai+anime+2025+with+release+date%2C+synopsis%2C+number+of+episode%2C+airing+status&format=json&pageno=1&safesearch=2&language=en&engines=google%2Cbrave%2Cduckduckgo%2Cbing%2Cyahoo
"""

expected = """
/
"Database" "Threads"

Searxng link: 
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


    result = remove_links(text_with_sample_links)
    logger.success(f"Result: '{result}'")

    save_file(result, f"{output_dir}/result.txt")
