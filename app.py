import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_chroma import Chroma

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

load_dotenv()

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Conversational RAG",
    layout="wide",
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f8fafc;
}

/* Hero Section */
.hero {
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
    
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    
    box-shadow: 0px 10px 25px rgba(
        37,99,235,0.2
    );
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #e5e7eb;
}

/* Metrics */
[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 10px rgba(
        0,0,0,0.05
    );
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #2563eb;
    border-radius: 15px;
    padding: 15px;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #e5e7eb;
}

/* Buttons */
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
}

.stButton button:hover {
    background-color: #1d4ed8;
}

/* Chat Input */
.stChatInputContainer {
    background: white;
    border-radius: 15px;
}

/* Headers */
h1,h2,h3 {
    color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>Conversational PDF Assistant</h1>
    <p>
        Chat with your PDFs using LangChain, Chroma, Groq and Conversational RAG
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = (
        InMemoryChatMessageHistory()
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "processed" not in st.session_state:
    st.session_state.processed = False

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    st.markdown("---")

    st.markdown("""
    ### Features

    ✅ Conversational RAG

    ✅ Chat Memory

    ✅ Groq LLM

    ✅ Chroma Vector DB

    ✅ Semantic Search
    """)

# ==================================================
# PROCESS PDF
# ==================================================

if uploaded_file and not st.session_state.processed:

    with st.spinner(
        "Processing PDF..."
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            pdf_path = temp_file.name

        # Load PDF

        loader = PyPDFLoader(
            pdf_path
        )

        docs = loader.load()

        # Split

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
        )

        chunks = (
            splitter.split_documents(
                docs
            )
        )

        # Embeddings

        embeddings = (
            HuggingFaceEmbeddings(
                model_name=
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        # Vector Store

        vectorstore = (
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
        )

        retriever = (
            vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
        )

        llm = ChatGroq(
            model=
            "llama-3.3-70b-versatile"
        )

        # --------------------------------------
        # History Aware Retriever
        # --------------------------------------

        contextualize_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
Given chat history and the latest user question,
rewrite the question into a standalone question.
Return only the rewritten question.
""",
                    ),
                    MessagesPlaceholder(
                        "chat_history"
                    ),
                    (
                        "human",
                        "{input}",
                    ),
                ]
            )
        )

        history_aware_retriever = (
            create_history_aware_retriever(
                llm,
                retriever,
                contextualize_prompt,
            )
        )

        # --------------------------------------
        # QA Prompt
        # --------------------------------------

        qa_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are an AI assistant.

Answer only using the
retrieved document context.

If the answer is not found,
say:

I could not find that information
in the uploaded PDF.

Context:
{context}
""",
                    ),
                    MessagesPlaceholder(
                        "chat_history"
                    ),
                    (
                        "human",
                        "{input}",
                    ),
                ]
            )
        )

        document_chain = (
            create_stuff_documents_chain(
                llm,
                qa_prompt,
            )
        )

        rag_chain = (
            create_retrieval_chain(
                history_aware_retriever,
                document_chain,
            )
        )

        st.session_state.rag_chain = (
            rag_chain
        )

        st.session_state.docs = docs
        st.session_state.chunks = chunks
        st.session_state.processed = True

    st.toast(
        "✅ PDF Processed Successfully!",
        icon="🎉"
    )

# ==================================================
# PDF STATS
# ==================================================

if st.session_state.processed:

    col1, col2 = st.columns(2)

    col1.metric(
    "📄 Total Pages",
    len(st.session_state.docs)
    )

    col2.metric(
        "🧩 Chunks Created",
        len(st.session_state.chunks)
    )

# ==================================================
# CHAT HISTORY DISPLAY
# ==================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

# ==================================================
# USER INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about the PDF..."
)

if (
    question
    and st.session_state.rag_chain
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner(
        "Generating answer..."
    ):

        response = (
            st.session_state.rag_chain.invoke(
                {
                    "input": question,
                    "chat_history":
                    st.session_state
                    .chat_history
                    .messages,
                }
            )
        )

        answer = response["answer"]

    with st.chat_message(
        "assistant"
    ):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.chat_history.add_message(
        HumanMessage(
            content=question
        )
    )

    st.session_state.chat_history.add_message(
        AIMessage(
            content=answer
        )
    )

    st.markdown("""
<hr>

<center>
<p style='color:gray'>
Built with ❤️ using LangChain, Chroma, Groq & Streamlit
</p>
</center>
""",
unsafe_allow_html=True)