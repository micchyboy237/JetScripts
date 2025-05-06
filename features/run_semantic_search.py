from jet.features.semantic_search import search_content

# Usage example
if __name__ == "__main__":
    query = 'List upcoming isekai anime this year (2024-2025).'
    content = """## Naruto: Shippuuden Movie 6 - Road to Ninja
Movie, 2012 Finished 1 ep, 109 min
Action Adventure Fantasy
Naruto: Shippuuden Movie 6 - Road to Ninja
Returning home to Konohagakure, the young ninja celebrate defeating a group of supposed Akatsuki members. Naruto Uzumaki and Sakura Haruno, however, feel differently. Naruto is jealous of his comrades' congratulatory families, wishing for the presence of his own parents. Sakura, on the other hand, is angry at her embarrassing parents, and wishes for no parents at all. The two clash over their opposing ideals, but are faced with a more pressing matter when the masked Madara Uchiha suddenly appears and transports them to an alternate world. In this world, Sakura's parents are considered heroes--for they gave their lives to protect Konohagakure from the Nine-Tailed Fox attack 10 years ago. Consequently, Naruto's parents, Minato Namikaze and Kushina Uzumaki, are alive and well. Unable to return home or find the masked Madara, Naruto and Sakura stay in this new world and enjoy the changes they have always longed for. All seems well for the two ninja, until an unexpected threat emerges that pushes Naruto and Sakura to not only fight for the Konohagakure of the alternate world, but also to find a way back to their own. [Written by MAL Rewrite]
Studio Pierrot
Source Manga
Theme Isekai
Demographic Shounen
7.68
366K
Add to My List"""

    result = search_content(
        query=query,
        content=content,
        threshold=0.2,
        top_k=None,
        min_length=25,
        max_length=300,
        max_result_tokens=300
    )

    print("\n=== Search Output Summary ===")
    print(f"Mean Pooling Results Count: {len(result['mean_pooling_results'])}")
    print(f"Mean Pooling Total Tokens: {result['mean_pooling_tokens']}")
    print(f"CLS Token Results Count: {len(result['cls_token_results'])}")
    print(f"CLS Token Total Tokens: {result['cls_token_tokens']}")
    print("\nMean Pooling Text:")
    print(result['mean_pooling_text'] or "No results")
    print("\nCLS Token Text:")
    print(result['cls_token_text'] or "No results")
