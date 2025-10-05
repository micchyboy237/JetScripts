import os
from jet.file.utils import save_file
from jet.wordnet.text_chunker import chunk_sentences

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

sample = "[ Resources ](/r/LangChain/?f=flair_name%3A%22Resources%22)\nI really liked this idea of evaluating different RAG strategies. This simple project is amazing and can be useful to the community here. You can have your custom data evaluate different RAG strategies and finally can see which one works best. Try and let me know what you guys think: [ https://www.ragarena.com/ ](https://www.ragarena.com/)\nPublic\nAnyone can view, post, and comment to thisck-the-right-method-and-why-your-ai-s-success-2cedcda99f8a&user=Mehulpratapsingh&userId=99320eff683e&source=---header_actions--2cedcda99f8a---------------------clap_footer------------------)\n\\--\n[ ](/ leverages both visual and textual information.\n4. SafeRAG: This paper talks covers the benchmark designed to evaluate the security vulnerabilities of RAG systems against adversarial attacks.\n5. Agentic RAG : This paper covers Agentic RAG, which is the fusion of RAG with agents, improving the retrieval process with decision-making and reasoning capabilities.\n6. TrustRAG: This is another paper that covers a security-focused framework designed to protect Retrieval-Augmented Generation (RAG) systems from corpus poisoning attacks.\n7. Enhancing RAG: Best Practices: This study explores key design factors influencing RAG systems, including query expansion, retrieval strategies, and In-Context Learning.\n8. Chain of Retrieval Augmented Generation: This paper covers the CoRG technique that improves RAG by iteratively retrieving and reasoning over the information before generating an answer.\n9. Fact, Fetch and Reason: This paper talks about a high-quality evaluation dataset called FRAMES, designed to evaluate LLMs' factuality, retrieval, and reasoning in end-to-end RAG scenarios.\n10. LONG 2 RAG: LONG 2 RAG is a new benchmark designed to evaluate RAG systems on long-context retrieval and long-form response generation."


if __name__ == "__main__":
    result = chunk_sentences(sample, chunk_size=2, chunk_overlap=1)
    save_file(result, f"{OUTPUT_DIR}/result.json")
