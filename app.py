import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from groq import Groq
from pathlib import Path
from langchain.schema import Document

# Set up Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key is None:
    st.error("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable in your Colab secrets.")
    st.stop()

# Check if PINECONE_ENVIRONMENT is needed and set it if necessary
# pinecone_env = os.environ.get("PINECONE_ENVIRONMENT") # Uncomment if needed
# if pinecone_env:
#     pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
# else:
try:
    pc = Pinecone(api_key=pinecone_api_key)
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

# Ensure the embedding model is passed correctly to PineconeVectorStore
try:
    vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=embeddings_model, namespace="https://github.com/CoderAgent/SecureAgent")
except Exception as e:
    st.error(f"Error initializing Pinecone Vector Store: {e}")
    st.stop()


# Set up Groq
def perform_rag(query):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key is None:
        # This should ideally be caught by the Streamlit app section, but adding here for redundancy
        return "Error: Groq API key not found. Please set the GROQ_API_KEY environment variable."

    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return "An error occurred while initializing the Groq client."


    try:
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
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        # Provide a more informative error message to the user
        return f"An error occurred while communicating with the Groq API: {e}"

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
