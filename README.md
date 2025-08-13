# Codebase RAG Sage 🌲✨

[![Streamlit](https://img.shields.io/badge/Streamlit-App-blue)](#)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-orange)](#)
[![LLM](https://img.shields.io/badge/Groq-LLaMA-8192-context-purple)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](#)

---

##  What Is Codebase RAG Sage?

**Codebase RAG Sage** is your AI-powered assistant for exploring and understanding codebases. It merges *Retrieval-Augmented Generation (RAG)* with **Streamlit**, **Pinecone**, and **Groq’s LLaMA models** to deliver contextual, code-aware answers to your queries.

### Key Features

- **Conversational Code Exploration** – Ask anything about your codebase in natural language and get informed responses grounded in actual source files.
- **Semantic Search** – Powered by embeddings (via `HuggingFaceEmbeddings`), for smarter, context-aware retrieval.
- **Namespace-Based Code Indexing** – Handle multiple repositories, each with a distinct namespace in Pinecone.
- **Streamlit Web Interface** – Elegant and intuitive UI for all your codebase queries.

---

##  Demo Snapshot

![App Demo Screenshot](screenshot.png)

*(Here you can show how the app looks—like the input field, cloned repo display, and Q&A section.)*

---

##  Quick Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/rahatmoktadir03/codebase-rag-sage.git
cd codebase-rag-sage
```

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies 

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
 - Create a .env file or export these:
   ```bash
   export PINECONE_API_KEY=your_pinecone_api_key
   export GROQ_API_KEY=your_groq_api_key
   ```

### 5. Create Pinecone Index (if needed)

```python
import pinecone
pinecone.init(api_key="YOUR_KEY")
pinecone.create_index("codebase-rag", dimension=768)
```
