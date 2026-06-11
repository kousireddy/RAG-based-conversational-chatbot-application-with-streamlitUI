# Conversational PDF Assistant

An AI-powered Conversational RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and interact with them through natural language conversations.

Built using **LangChain**, **ChromaDB**, **Groq LLM**, **HuggingFace Embeddings**, and **Streamlit**, the application retrieves relevant information from uploaded PDFs and generates accurate, context-aware answers while maintaining conversation history.

---

## Features

*  Upload and process PDF documents
*  Conversational question answering
*  Chat history-aware retrieval
*  Semantic search using vector embeddings
*  Fast response generation with Groq LLM
*  Retrieval-Augmented Generation (RAG)
*  Chroma Vector Database integration
*  Context-grounded answers
*  Document statistics (Pages & Chunks)
*  Interactive Streamlit UI

---

##  Architecture

```text
PDF Upload
    │
    ▼
PyPDFLoader
    │
    ▼
Text Splitting
    │
    ▼
Embeddings Generation
    │
    ▼
Chroma Vector Store
    │
    ▼
Retriever
    │
    ▼
History-Aware Retrieval
    │
    ▼
Groq LLM
    │
    ▼
Generated Answer
```

---

##  Tech Stack

| Component             | Technology                     |
| --------------------- | ------------------------------ |
| Frontend              | Streamlit                      |
| Framework             | LangChain                      |
| LLM                   | Groq (Llama 3.3 70B Versatile) |
| Vector Database       | ChromaDB                       |
| Embedding Model       | all-MiniLM-L6-v2               |
| PDF Processing        | PyPDFLoader                    |
| Text Splitting        | RecursiveCharacterTextSplitter |
| Memory                | InMemoryChatMessageHistory     |
| Environment Variables | python-dotenv                  |

---

##  Project Structure

```text
Conversational-PDF-Assistant/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
└── assets/
```

---

##  Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Conversational-PDF-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶ Run the Application

```bash
streamlit run app.py
```

After running, open:

```text
http://localhost:8501
```

---

##  Workflow

### Step 1: Upload PDF

The user uploads a PDF document through the Streamlit interface.

### Step 2: Extract Text

PyPDFLoader extracts text from the PDF.

### Step 3: Chunking

The document is divided into smaller chunks using:

```python
chunk_size=1000
chunk_overlap=200
```

### Step 4: Generate Embeddings

Each chunk is converted into vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

### Step 5: Store Embeddings

Embeddings are stored inside ChromaDB.

### Step 6: Retrieve Context

Relevant chunks are retrieved based on semantic similarity.

### Step 7: History-Aware Retrieval

Follow-up questions are converted into standalone questions before retrieval.

### Step 8: Generate Response

Retrieved context is passed to the Groq LLM to generate answers.

---

##  LangChain Components Used

### PyPDFLoader

Loads PDF documents.

### RecursiveCharacterTextSplitter

Splits documents into manageable chunks.

### HuggingFaceEmbeddings

Generates vector embeddings.

### Chroma

Stores embeddings and enables semantic retrieval.

### create_history_aware_retriever

Handles conversational context.

### create_stuff_documents_chain

Injects retrieved context into prompts.

### create_retrieval_chain

Combines retrieval and answer generation.

---

##  Application Features

* Conversational PDF Chat
* Context-Aware Retrieval
* Memory-Based Conversations
* Semantic Search
* Document Statistics
* Modern Streamlit UI
* Fast Inference using Groq

---

##  Example Questions

After uploading a PDF, users can ask:

* What is the main topic of the document?
* Summarize the document.
* What are the key findings?
* Who is the author?
* Explain chapter 3.
* What conclusions are mentioned?

---

##  Future Enhancements

* Multiple PDF Support
* Source Citations
* Persistent Chroma Storage
* Hybrid Search (BM25 + Vector Search)
* Authentication & User Accounts
* Voice-Based Querying
* Streaming Responses
* Multi-LLM Support
* Cloud Deployment

---

##  Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* LangChain Framework
* Vector Databases
* Semantic Search
* Prompt Engineering
* Conversational AI
* LLM Integration
* Streamlit Development
* Memory Management
* Document Question Answering

---

##  Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

##  License

This project is licensed under the MIT License.

---

##  Author

Developed using LangChain, ChromaDB, Groq, HuggingFace Embeddings, and Streamlit to build an end-to-end Conversational RAG application for PDF Question Answering.
