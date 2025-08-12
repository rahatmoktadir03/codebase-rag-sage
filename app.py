import streamlit as st
from sentence_transformers import SentenceTransformer
# Correct the import based on potential library changes or usage
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from groq import Groq
from pathlib import Path
from langchain.schema import Document

# Set up Pinecone
# It's good practice to check if the environment variable is set
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# Check if PINECONE_ENVIRONMENT is needed and set it if necessary
# pinecone_env = os.environ.get("PINECONE_ENVIRONMENT") # Uncomment if needed
# if pinecone_env:
#     pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
# else:
pc = Pinecone(api_key=pinecone_api_key)

pinecone_index = pc.Index("codebase-rag")

# Set up Embeddings
@st.cache_resource
def get_huggingface_embeddings(model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model

embeddings_model = get_huggingface_embeddings()

# Ensure the embedding model is passed correctly to PineconeVectorStore
vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=embeddings_model, namespace="https://github.com/CoderAgent/SecureAgent")


# Set up Groq
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

def perform_rag(query):
    raw_query_embedding = embeddings_model.encode(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace="https://github.com/CoderAgent/SecureAgent")

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

# Streamlit App
st.title("Codebase RAG Project")

query = st.text_input("Enter your query about the codebase:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Getting your answer..."):
            response = perform_rag(query)
            st.write(response)
    else:
        st.warning("Please enter a query.")
