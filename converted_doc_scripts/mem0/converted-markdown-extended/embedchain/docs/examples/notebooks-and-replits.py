from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Notebooks & Replits
---

# Explore awesome apps

Check out the remarkable work accomplished using [Embedchain](https://app.embedchain.ai/custom-gpts/).

## Collection of Google colab notebook and Replit links for users

Get started with Embedchain by trying out the examples below. You can run the examples in your browser using Google Colab or Replit.

<table>
  <thead>
    <tr>
      <th>LLM</th>
      <th>Google Colab</th>
      <th>Replit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td className="align-middle">MLX</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/openai.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/openai#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Anthropic</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/anthropic.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/anthropic#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Azure MLX</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/azure-openai.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/azureopenai#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">VertexAI</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/vertex_ai.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/vertexai#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Cohere</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/cohere.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/cohere#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Together</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/together.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Ollama</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/ollama.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Hugging Face</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/hugging_face_hub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/huggingface#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">JinaChat</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/jina.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/jina#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">GPT4All</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/gpt4all.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/gpt4all#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Llama2</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/llama2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/llama2#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
  </tbody>
</table>
<table>
  <thead>
    <tr>
      <th>Embedding model</th>
      <th>Google Colab</th>
      <th>Replit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td className="align-middle">MLX</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/openai.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/openai#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">VertexAI</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/vertex_ai.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/vertexai#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">GPT4All</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/gpt4all.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/gpt4all#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Hugging Face</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/hugging_face_hub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/huggingface#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
  </tbody>
</table>
<table>
  <thead>
    <tr>
      <th>Vector DB</th>
      <th>Google Colab</th>
      <th>Replit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td className="align-middle">ChromaDB</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/chromadb.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/chromadb#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Elasticsearch</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/elasticsearch.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/elasticsearchdb#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Opensearch</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/opensearch.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/opensearchdb#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
    <tr>
      <td className="align-middle">Pinecone</td>
      <td className="align-middle"><a target="_blank" href="https://colab.research.google.com/github/embedchain/embedchain/blob/main/notebooks/pinecone.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" noZoom alt="Open In Colab"/></a></td>
      <td className="align-middle"><a target="_blank" href="https://replit.com/@taranjeetio/pineconedb#main.py"><img src="https://replit.com/badge?caption=Try%20with%20Replit&amp;variant=small" noZoom alt="Try with Replit Badge"/></a></td>
    </tr>
  </tbody>
</table>
"""
logger.info("# Explore awesome apps")

logger.info("\n\n[DONE]", bright=True)