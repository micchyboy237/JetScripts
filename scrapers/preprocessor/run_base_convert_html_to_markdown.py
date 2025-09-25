import os
import shutil

from jet.scrapers.preprocessor import base_convert_html_to_markdown
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

sample_html = """
<pre class="wp-block-prismatic-blocks language-python" datatext="" tabindex="0">
	<code class="language-python">
		<span class="token keyword">import</span> pandas 
		<span class="token keyword">as</span> pd

		<span class="token keyword">import</span> requests


		<span class="token keyword">def</span>
		<span class="token function">import_data</span>
		<span class="token punctuation">(</span>pages
		<span class="token punctuation">,</span> start_year
		<span class="token punctuation">,</span> end_year
		<span class="token punctuation">,</span> search_terms
		<span class="token punctuation">)</span>
		<span class="token punctuation">:</span>
		<span class="token triple-quoted-string string">\"\"\"
    This function is used to use the OpenAlex API, conduct a search on works, a return a dataframe with associated works.
    
    Inputs: 
        - pages: int, number of pages to loop through
        - search_terms: str, keywords to search for (must be formatted according to OpenAlex standards)
        - start_year and end_year: int, years to set as a range for filtering works
    \"\"\"</span>
		<span class="token comment">#create an empty dataframe</span>
    search_results 
		<span class="token operator">=</span> pd
		<span class="token punctuation">.</span>DataFrame
		<span class="token punctuation">(</span>
		<span class="token punctuation">)</span>
		<span class="token keyword">for</span> page 
		<span class="token keyword">in</span>
		<span class="token builtin">range</span>
		<span class="token punctuation">(</span>
		<span class="token number">1</span>
		<span class="token punctuation">,</span> pages
		<span class="token punctuation">)</span>
		<span class="token punctuation">:</span>
		<span class="token comment">#use paramters to conduct request and format to a dataframe</span>
        response 
		<span class="token operator">=</span> requests
		<span class="token punctuation">.</span>get
		<span class="token punctuation">(</span>
		<span class="token string-interpolation">
			<span class="token string">f'https://api.openalex.org/works?page=</span>
			<span class="token interpolation">
				<span class="token punctuation">{</span>page
				<span class="token punctuation">}</span>
			</span>
			<span class="token string">&amp;per-page=200&amp;filter=publication_year:</span>
			<span class="token interpolation">
				<span class="token punctuation">{</span>start_year
				<span class="token punctuation">}</span>
			</span>
			<span class="token string">-</span>
			<span class="token interpolation">
				<span class="token punctuation">{</span>end_year
				<span class="token punctuation">}</span>
			</span>
			<span class="token string">,type:article&amp;search=</span>
			<span class="token interpolation">
				<span class="token punctuation">{</span>search_terms
				<span class="token punctuation">}</span>
			</span>
			<span class="token string">'</span>
		</span>
		<span class="token punctuation">)</span>
        data 
		<span class="token operator">=</span> pd
		<span class="token punctuation">.</span>DataFrame
		<span class="token punctuation">(</span>response
		<span class="token punctuation">.</span>json
		<span class="token punctuation">(</span>
		<span class="token punctuation">)</span>
		<span class="token punctuation">[</span>
		<span class="token string">'results'</span>
		<span class="token punctuation">]</span>
		<span class="token punctuation">)</span>
		<span class="token comment">#append to empty dataframe</span>
        search_results 
		<span class="token operator">=</span> pd
		<span class="token punctuation">.</span>concat
		<span class="token punctuation">(</span>
		<span class="token punctuation">[</span>search_results
		<span class="token punctuation">,</span> data
		<span class="token punctuation">]</span>
		<span class="token punctuation">)</span>
		<span class="token comment">#subset to relevant features</span>
    search_results 
		<span class="token operator">=</span> search_results
		<span class="token punctuation">[</span>
		<span class="token punctuation">[</span>
		<span class="token string">"id"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"title"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"display_name"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"publication_year"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"publication_date"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"type"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"countries_distinct_count"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"institutions_distinct_count"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"has_fulltext"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"cited_by_count"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"keywords"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"referenced_works_count"</span>
		<span class="token punctuation">,</span>
		<span class="token string">"abstract_inverted_index"</span>
		<span class="token punctuation">]</span>
		<span class="token punctuation">]</span>
		<span class="token keyword">return</span>
		<span class="token punctuation">(</span>search_results
		<span class="token punctuation">)</span>
		<span class="token comment">#search for AI-related research</span>
ai_search 
		<span class="token operator">=</span> import_data
		<span class="token punctuation">(</span>
		<span class="token number">30</span>
		<span class="token punctuation">,</span>
		<span class="token number">2018</span>
		<span class="token punctuation">,</span>
		<span class="token number">2025</span>
		<span class="token punctuation">,</span>
		<span class="token string">"'artificial intelligence' OR 'deep learn' OR 'neural net' OR 'natural language processing' OR 'machine learn' OR 'large language models' OR 'small language models'"</span>
		<span class="token punctuation">)</span>
	</code>
</pre>
"""



if __name__ == "__main__":
    markdown_text = base_convert_html_to_markdown(sample_html)
    save_file(markdown_text, f"{OUTPUT_DIR}/markdown.md")
