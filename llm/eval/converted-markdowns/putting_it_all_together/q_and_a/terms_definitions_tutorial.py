from llama_index.readers.file import ImageReader
from PIL import Image
from constants import REFINE_TEMPLATE, TEXT_QA_TEMPLATE
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core import (
    PromptTemplate,
    SelectorPromptTemplate,
    ChatPromptTemplate,
)
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core import Settings
from jet.llm.ollama.base import Ollama
from llama_index.core import Document, SummaryIndex, load_index_from_storage
import os
import streamlit as st
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()


st.title("🦙 Llama Index Term Extractor 🦙")

document_text = st.text_area("Enter raw text")
if st.button("Extract Terms and Definitions") and document_text:
    with st.spinner("Extracting..."):
        extracted_terms = document_text  # this is a placeholder!
    st.write(extracted_terms)


DEFAULT_TERM_STR = (
    "Make a list of terms and definitions that are defined in the context, "
    "with one pair on each line. "
    "If a term is missing it's definition, use your best judgment. "
    "Write each line as as follows:\nTerm: <term> Definition: <definition>"
)

st.title("🦙 Llama Index Term Extractor 🦙")

setup_tab, upload_tab = st.tabs(["Setup", "Upload/Extract Terms"])

with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your Ollama API key here", type="password")
    llm_name = st.selectbox("Which LLM?", ["gpt-3.5-turbo", "gpt-4"])
    model_temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, step=0.1
    )
    term_extract_str = st.text_area(
        "The query to extract terms and definitions with.",
        value=DEFAULT_TERM_STR,
    )

with upload_tab:
    st.subheader("Extract and Query Definitions")
    document_text = st.text_area("Enter raw text")
    if st.button("Extract Terms and Definitions") and document_text:
        with st.spinner("Extracting..."):
            extracted_terms = document_text  # this is a placeholder!
        st.write(extracted_terms)


def get_llm(llm_name, model_temperature, api_key, max_tokens=256):
    #     os.environ["OPENAI_API_KEY"] = api_key
    return Ollama(
        temperature=model_temperature, model=llm_name, max_tokens=max_tokens
    )


def extract_terms(
    documents, term_extract_str, llm_name, model_temperature, api_key
):
    llm = get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

    temp_index = SummaryIndex.from_documents(
        documents,
    )
    query_engine = temp_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )
    terms_definitions = str(query_engine.query(term_extract_str))
    terms_definitions = [
        x
        for x in terms_definitions.split("\n")
        if x and "Term:" in x and "Definition:" in x
    ]
    terms_to_definition = {
        x.split("Definition:")[0]
        .split("Term:")[-1]
        .strip(): x.split("Definition:")[-1]
        .strip()
        for x in terms_definitions
    }
    return terms_to_definition


...
with upload_tab:
    st.subheader("Extract and Query Definitions")
    document_text = st.text_area("Enter raw text")
    if st.button("Extract Terms and Definitions") and document_text:
        with st.spinner("Extracting..."):
            extracted_terms = extract_terms(
                [Document(text=document_text)],
                term_extract_str,
                llm_name,
                model_temperature,
                api_key,
            )
        st.write(extracted_terms)


...
if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS
...


def insert_terms(terms_to_definition):
    for term, definition in terms_to_definition.items():
        doc = Document(text=f"Term: {term}\nDefinition: {definition}")
        st.session_state["llama_index"].insert(doc)


@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Create the VectorStoreIndex object."""
    Settings.llm = get_llm(llm_name, model_temperature, api_key)

    index = VectorStoreIndex([])

    return index, llm


...

with upload_tab:
    st.subheader("Extract and Query Definitions")
    if st.button("Initialize Index and Reset Terms"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = {}

    if "llama_index" in st.session_state:
        st.markdown(
            "Either upload an image/screenshot of a document, or enter the text manually."
        )
        document_text = st.text_area("Or enter raw text")
        if st.button("Extract Terms and Definitions") and (
            uploaded_file or document_text
        ):
            st.session_state["terms"] = {}
            terms_docs = {}
            with st.spinner("Extracting..."):
                terms_docs.update(
                    extract_terms(
                        [Document(text=document_text)],
                        term_extract_str,
                        llm_name,
                        model_temperature,
                        api_key,
                    )
                )
            st.session_state["terms"].update(terms_docs)

        if "terms" in st.session_state and st.session_state["terms"]:
            st.markdown("Extracted terms")
            st.json(st.session_state["terms"])

            if st.button("Insert terms?"):
                with st.spinner("Inserting terms"):
                    insert_terms(st.session_state["terms"])
                st.session_state["all_terms"].update(st.session_state["terms"])
                st.session_state["terms"] = {}
                st.experimental_rerun()

...
setup_tab, terms_tab, upload_tab, query_tab = st.tabs(
    ["Setup", "All Terms", "Upload/Extract Terms", "Query Terms"]
)
...
with terms_tab:
    with terms_tab:
        st.subheader("Current Extracted Terms and Definitions")
        st.json(st.session_state["all_terms"])
...
with query_tab:
    st.subheader("Query for Terms/Definitions!")
    st.markdown(
        (
            "The LLM will attempt to answer your query, and augment it's answers using the terms/definitions you've inserted. "
            "If a term is not in the index, it will answer using it's internal knowledge."
        )
    )
    if st.button("Initialize Index and Reset Terms", key="init_index_2"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = {}

    if "llama_index" in st.session_state:
        query_text = st.text_input("Ask about a term or definition:")
        if query_text:
            query_text = (
                query_text
                + "\nIf you can't find the answer, answer the query with the best of your knowledge."
            )
            with st.spinner("Generating answer..."):
                response = (
                    st.session_state["llama_index"]
                    .as_query_engine(
                        similarity_top_k=5,
                        response_mode="compact",
                        text_qa_template=TEXT_QA_TEMPLATE,
                        refine_template=DEFAULT_REFINE_PROMPT,
                    )
                    .query(query_text)
                )
            st.markdown(str(response))


def insert_terms(terms_to_definition):
    for term, definition in terms_to_definition.items():
        doc = Document(text=f"Term: {term}\nDefinition: {definition}")
        st.session_state["llama_index"].insert(doc)
    st.session_state["llama_index"].storage_context.persist()


@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Load the Index object."""
    Settings.llm = get_llm(llm_name, model_temperature, api_key)

    index = load_index_from_storage(storage_context)

    return index


...
if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS
...


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information answer the following question "
    "(if you don't know the answer, use the best of your knowledge): {query_str}\n"
)
TEXT_QA_TEMPLATE = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context and using the best of your knowledge, improve the existing answer. "
    "If you can't improve the existing answer, just repeat it again."
)
DEFAULT_REFINE_PROMPT = PromptTemplate(DEFAULT_REFINE_PROMPT_TMPL)

CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content="We have the opportunity to refine the above answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context and using the best of your knowledge, improve the existing answer. "
        "If you can't improve the existing answer, just repeat it again.",
        role=MessageRole.USER,
    ),
]

CHAT_REFINE_PROMPT = ChatPromptTemplate(CHAT_REFINE_PROMPT_TMPL_MSGS)

REFINE_TEMPLATE = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_PROMPT,
    conditionals=[(is_chat_model, CHAT_REFINE_PROMPT)],
)


...
if "llama_index" in st.session_state:
    query_text = st.text_input("Ask about a term or definition:")
    if query_text:
        query_text = query_text  # Notice we removed the old instructions
        with st.spinner("Generating answer..."):
            response = (
                st.session_state["llama_index"]
                .as_query_engine(
                    similarity_top_k=5,
                    response_mode="compact",
                    text_qa_template=TEXT_QA_TEMPLATE,
                    refine_template=DEFAULT_REFINE_PROMPT,
                )
                .query(query_text)
            )
        st.markdown(str(response))
...


@st.cache_resource
def get_file_extractor():
    image_parser = ImageReader(keep_image=True, parse_text=True)
    file_extractor = {
        ".jpg": image_parser,
        ".png": image_parser,
        ".jpeg": image_parser,
    }
    return file_extractor


file_extractor = get_file_extractor()
...
with upload_tab:
    st.subheader("Extract and Query Definitions")
    if st.button("Initialize Index and Reset Terms", key="init_index_1"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = DEFAULT_TERMS

    if "llama_index" in st.session_state:
        st.markdown(
            "Either upload an image/screenshot of a document, or enter the text manually."
        )
        uploaded_file = st.file_uploader(
            "Upload an image/screenshot of a document:",
            type=["png", "jpg", "jpeg"],
        )
        document_text = st.text_area("Or enter raw text")
        if st.button("Extract Terms and Definitions") and (
            uploaded_file or document_text
        ):
            st.session_state["terms"] = {}
            terms_docs = {}
            with st.spinner("Extracting (images may be slow)..."):
                if document_text:
                    terms_docs.update(
                        extract_terms(
                            [Document(text=document_text)],
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
                if uploaded_file:
                    Image.open(uploaded_file).convert("RGB").save("temp.png")
                    img_reader = SimpleDirectoryReader(
                        input_files=["temp.png"], file_extractor=file_extractor
                    )
                    img_docs = img_reader.load_data()
                    os.remove("temp.png")
                    terms_docs.update(
                        extract_terms(
                            img_docs,
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
            st.session_state["terms"].update(terms_docs)

        if "terms" in st.session_state and st.session_state["terms"]:
            st.markdown("Extracted terms")
            st.json(st.session_state["terms"])

            if st.button("Insert terms?"):
                with st.spinner("Inserting terms"):
                    insert_terms(st.session_state["terms"])
                st.session_state["all_terms"].update(st.session_state["terms"])
                st.session_state["terms"] = {}
                st.experimental_rerun()

logger.info("\n\n[DONE]", bright=True)
