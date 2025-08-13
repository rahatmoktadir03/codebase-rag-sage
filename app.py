import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from groq import Groq
from pathlib import Path
from langchain.schema import Document
from git import Repo
import shutil
import tempfile # Import tempfile

# Helper function to clone repository (from previous cells)
def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    # Use a more robust temporary directory approach
    temp_dir = Path(tempfile.mkdtemp())
    repo_path = temp_dir / repo_name

    # Remove the existing directory if it's not empty
    if repo_path.exists() and repo_path.is_dir():
        shutil.rmtree(repo_path)

    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

# Helper function to get file content (from previous cells)
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

def get_file_content(file_path, repo_path):
    """
    Get content of a single file.

    Args:
        file_path (str): Path to the file

    Returns:
        Optional[Dict[str, str]]: Dictionary with file name and content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get relative path from repo root
        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None


def get_main_files_content(repo_path: str):
    """
    Get content of supported code files from the local repository.

    Args:
        repo_path: Path to the local repository

    Returns:
        List of dictionaries containing file names and contents
    """
    files_content = []

    try:
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content


# Set up Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key is None:
    st.error("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable in your Colab secrets.")
    st.stop()

try:
    # Specify the environment here
    pc = Pinecone(api_key=pinecone_api_key, environment="gcp-starter")
    # Check if index exists before connecting
    if "codebase-rag" not in pc.list_indexes():
        st.error("Pinecone index 'codebase-rag' not found. Please create it in your Pinecone dashboard with dimension 768.")
        st.stop()
    pinecone_index = pc.Index("codebase-rag")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()


# Set up Embeddings
@st.cache_resource
def get_huggingface_embeddings(model_name="sentence-transformers/all-mpnet-base-v2"):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        st.stop()

embeddings_model = get_huggingface_embeddings()

# Set up Groq
def perform_rag(query, namespace):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key is None:
        return "Error: Groq API key not found. Please set the GROQ_API_KEY environment variable."

    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return "An error occurred while initializing the Groq client."


    try:
        raw_query_embedding = embeddings_model.encode(query)

        # Reduce top_k to retrieve fewer contexts
        top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)

        # Get the list of retrieved texts
        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.
        Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
        """

        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192", # Or your preferred Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return f"An error occurred while communicating with the Groq API: {e}"

# Streamlit App
st.title("Codebase RAG Project")

# Repository Cloning Section
st.header("Clone GitHub Repository")
repo_url = st.text_input("Enter GitHub Repository URL:", key="repo_url_input")
clone_button = st.button("Clone Repository")

if clone_button and repo_url:
    with st.spinner("Cloning repository and processing files..."):
        try:
            repo_path = clone_repository(repo_url)
            file_content = get_main_files_content(repo_path)

            # Use the repo URL as the namespace
            namespace = repo_url

            # Add documents to Pinecone
            documents = []
            for file in file_content:
                doc = Document(
                    page_content=f"{file['name']}\n{file['content']}",
                    metadata={"source": file['name']}
                )
                documents.append(doc)

            # Delete existing namespace if it exists
            try:
                pinecone_index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                print(f"Error deleting existing namespace {namespace}: {e}")
                # Continue even if deletion fails, as the namespace might not exist

            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings_model,
                index_name="codebase-rag",
                namespace=namespace
            )

            st.success(f"Successfully cloned and processed repository: {repo_url}")

            # Store repo_path and namespace in session state for later use
            st.session_state['repo_path'] = repo_path
            st.session_state['namespace'] = namespace
            st.session_state['file_content'] = file_content # Store file content to potentially display

        except Exception as e:
            st.error(f"Error cloning or processing repository: {e}")

# Display Processed File Content (Optional, based on screenshots)
if 'file_content' in st.session_state and st.session_state['file_content']:
    st.header("Processed Files")
    # You can choose to display file names or a summary
    file_names = [file['name'] for file in st.session_state['file_content']]
    st.write(f"Processed {len(file_names)} supported files.")
    # Optionally, display file names in an expander
    with st.expander("View Processed File Names"):
        st.write(file_names)


# RAG Query Section
st.header("Ask a Question about the Codebase")

# Check if a repository has been successfully processed
if 'namespace' in st.session_state:
    query = st.text_input("Enter your query:", key="query_input")
    get_answer_button = st.button("Get Answer")

    if get_answer_button and query:
        with st.spinner("Getting your answer..."):
            # Retrieve namespace from session state
            namespace = st.session_state['namespace']
            response = perform_rag(query, namespace)
            st.subheader("Answer:")
            st.write(response)
    elif get_answer_button and not query:
        st.warning("Please enter a query.")

else:
    st.info("Please clone a GitHub repository first to enable the RAG functionality.")
