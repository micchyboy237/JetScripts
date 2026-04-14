### Quick Start & Usage Examples

1. **Setup**
   ```bash
   mkdir my-wiki && cd my-wiki
   # Save the code above as llm_wiki.py
   python llm_wiki.py init
   export API_KEY=your_xai_or_openai_key   # or XAI_API_KEY
   ```

2. **Add your first source**
   ```bash
   # Put any markdown/text file in raw/
   cp ~/Downloads/research-paper.md raw/
   python llm_wiki.py ingest research-paper.md
   ```

3. **Ask questions**
   ```bash
   python llm_wiki.py query "What are the main contradictions in the sources about climate policy?"
   # or save the answer permanently:
   python llm_wiki.py query "Summarize the key themes across all entities" --save
   ```

4. **Periodic maintenance**
   ```bash
   python llm_wiki.py lint
   ```

5. **Real workflow (recommended)**
   - Keep Obsidian open on the `wiki/` folder
   - Use the graph view to see connections grow
   - Ingest one source at a time while you read the updates live
   - File interesting query results with `--save`

The pipeline exactly follows the documentation you provided: the LLM does all the bookkeeping, the wiki compounds, and you stay in control of sourcing and direction. Everything is version-controlled if you `git init` the folder. Enjoy your personal Memex!