import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from groq import Groq
from pathlib import Path
from git import Repo
import shutil
import tempfile

# ==============================
# Helper: Clone GitHub Repo
# ==============================
def clone_repository(repo_url):
    repo_name = repo_url.split("/")[-1]
    temp_dir = Path(tempfile.mkdtemp())
    repo_path = temp_dir / repo_name

    if repo_path.exists() and repo_path.is_dir():
        shutil.rmtree(repo_path)

    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

# ==============================
# Helper: Get File Content
# ==============================
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
    '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'
}
IGNORED_DIRS = {
    'node_modules', 'venv', 'env', 'dist', 'build', '.git',
    '__pycache__', '.next', '.vscode', 'vendor'
}

def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": content}
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        print(f"Error reading repository: {str(e)}")
    return files_content

# ==============================
# Pinecone Setup
# ==============================
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key is None:
    st.error("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")
    st.stop()

try:
    pc = Pinecone(api_key=pinecone_api_key)

    index_names = [idx.name for idx in pc.list_indexes()]
    if "codebase-rag" not in index_names:
        st.error("Pinecone index 'codebase-rag' not found. Please create it with dimension 768.")
        st.stop()

    pinecone_index = pc.Index("codebase-rag")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# ==============================
# HuggingFace Embeddings (LangChain)
# ==============================
@st.cache_resource
def get_huggingface_embeddings(model_name="sentence-transformers/all-mpnet-base-v2"):
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"Error loading HuggingFace embeddings: {e}")
        st.stop()

embeddings_model = get_huggingface_embeddings()

# ==============================
# Perform RAG
# ==============================
def perform_rag(query, namespace):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key is None:
        return "Error: Groq API key not found."

    try:
        client = Groq(api_key=groq_api_key)
        raw_query_embedding = embeddings_model.embed_query(query)

        top_matches = pinecone_index.query(
            vector=raw_query_embedding,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )

        contexts = [item['metadata']['text'] for item in top_matches['matches']]
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n" + query

        system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.
        Answer questions about the codebase based on the context provided.
        """

        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error during Groq API call: {e}"

# ==============================
# Streamlit App UI
# ==============================
st.title("Codebase RAG Project")

# Clone Repo Section
st.header("Clone GitHub Repository")
repo_url = st.text_input("Enter GitHub Repository URL:", key="repo_url_input")
clone_button = st.button("Clone Repository")

if clone_button and repo_url:
    with st.spinner("Cloning repository and processing files..."):
        try:
            repo_path = clone_repository(repo_url)
            file_content = get_main_files_content(repo_path)
            namespace = repo_url

            documents = [
                Document(
                    page_content=f"{file['name']}\n{file['content']}",
                    metadata={"source": file['name'], "text": file['content']}
                )
                for file in file_content
            ]

            try:
                pinecone_index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                print(f"Namespace deletion error (might not exist yet): {e}")

            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings_model,
                index_name="codebase-rag",
                namespace=namespace
            )

            st.success(f"Successfully cloned and processed: {repo_url}")
            st.session_state['repo_path'] = repo_path
            st.session_state['namespace'] = namespace
            st.session_state['file_content'] = file_content

        except Exception as e:
            st.error(f"Error cloning or processing repository: {e}")

# Show Processed Files
if 'file_content' in st.session_state and st.session_state['file_content']:
    st.header("Processed Files")
    file_names = [file['name'] for file in st.session_state['file_content']]
    st.write(f"Processed {len(file_names)} supported files.")
    with st.expander("View Processed File Names"):
        st.write(file_names)

# RAG Query Section
st.header("Ask a Question about the Codebase")
if 'namespace' in st.session_state:
    query = st.text_input("Enter your query:", key="query_input")
    get_answer_button = st.button("Get Answer")
    if get_answer_button and query:
        with st.spinner("Getting your answer..."):
            namespace = st.session_state['namespace']
            response = perform_rag(query, namespace)
            st.subheader("Answer:")
            st.write(response)
    elif get_answer_button and not query:
        st.warning("Please enter a query.")
else:
    st.info("Please clone a repository first to enable RAG.")
