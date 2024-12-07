
import streamlit as st
from groq import Groq
import os
from google.colab import userdata # Import the userdata module


st.title("ChatGPT-like clone (using Groq)")

# Initialize Groq client
groq_api_key = userdata.get("GROQ_API_KEY")  
client = Groq(api_key=groq_api_key) 

if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama3-8b-8192"  

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input handling
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send query to Groq and get response
    with st.chat_message("assistant"):
        response = client.chat_completions.create( 
            model=st.session_state["groq_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True, 
        )

        # If response is streamed, handle appropriately
        if isinstance(response, list):  # Assuming list-like responses for streamed output
            for chunk in response:
                st.write(chunk)
            final_response = "".join(response)
        else:
            final_response = response  # Non-streamed response
        
        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
