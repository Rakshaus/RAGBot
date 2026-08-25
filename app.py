import streamlit as st

from document_loader import load_document
from text_splitter import split_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore
from llm import create_llm
from rag_pipeline import generate_answer


st.set_page_config(
    page_title="RAGBot",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 RAGBot")

st.caption(
    "Your AI assistant powered by RAG"
)


@st.cache_resource(show_spinner=False)
def initialize_rag():

    documents = load_document(
        "sample.txt"
    )

    chunks = split_documents(
        documents
    )

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        embedding_model
    )

    vector_store.add_documents(
        chunks
    )

    tokenizer, model = create_llm()

    return (
        vector_store,
        tokenizer,
        model
    )


with st.spinner("Starting RAGBot..."):

    try:

        (
            vector_store,
            tokenizer,
            model
        ) = initialize_rag()

    except Exception as error:

        st.error(
            "RAGBot could not start."
        )

        st.exception(error)

        st.stop()


# -----------------------------------------
# Chat history
# -----------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
                "Hi! I'm RAGBot 🤖 How can I help you?"
        }
    ]


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# -----------------------------------------
# User input
# -----------------------------------------

question = st.chat_input(
    "Ask me anything..."
)


if question:

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -------------------------------------
    # Search knowledge base
    # -------------------------------------

    search_results = vector_store.search(
        question,
        k=3
    )


    # -------------------------------------
    # IMPORTANT:
    # Only use documents that are actually
    # relevant to the question.
    # -------------------------------------

    relevant_documents = []

    for document, distance in search_results:

        # Lower distance = more similar.
        #
        # 0.8 is intentionally stricter than
        # the previous 1.2 threshold.

        if distance <= 0.8:

            relevant_documents.append(
                (
                    document,
                    distance
                )
            )


    # -------------------------------------
    # Generate answer
    # -------------------------------------

    with st.spinner("Thinking..."):

        try:

            answer = generate_answer(
                tokenizer,
                model,
                question,
                relevant_documents
            )

        except Exception as error:

            answer = (
                "Sorry, I couldn't generate "
                "a response."
            )

            st.error(str(error))


    # -------------------------------------
    # Display answer
    # -------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # -------------------------------------
    # Show RAG information
    # -------------------------------------

    if relevant_documents:

        with st.expander(
            "📚 Retrieved Information"
        ):

            for document, distance in (
                relevant_documents
            ):

                st.write(
                    "**Source:** "
                    + document.metadata.get(
                        "source",
                        "Unknown"
                    )
                )

                st.write(
                    document.page_content
                )

                st.caption(
                    f"FAISS distance: {distance:.4f}"
                )