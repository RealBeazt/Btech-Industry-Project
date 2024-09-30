import os
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

sec_key = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

st.title("Legal Assistant Bot")

if 'all_chats' not in st.session_state:
    st.session_state.all_chats = {}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = None

st.sidebar.title("Chat Sessions")

st.markdown("""
    <style>
    .chat-item {
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f0f5;
        border-radius: 5px;
        cursor: pointer;
        font-family: Arial, sans-serif;
    }
    .chat-item:hover {
        background-color: #e0e0eb;
    }
    .selected-chat {
        background-color: #d9d9e6;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

new_chat_title = st.sidebar.text_input("Enter a title for the new chat")

if st.sidebar.button("Start New Chat"):
    if new_chat_title and new_chat_title not in st.session_state.all_chats:
        st.session_state.all_chats[new_chat_title] = []
        st.session_state.current_chat = new_chat_title
        new_chat_title = ""
    else:
        st.sidebar.error("Chat title must be unique and not empty.")

st.sidebar.markdown("### Previous Chats")

for chat_title in st.session_state.all_chats.keys():
    chat_class = "selected-chat" if chat_title == st.session_state.current_chat else "chat-item"
    if st.sidebar.button(chat_title, key=chat_title):
        st.session_state.current_chat = chat_title

current_chat = st.session_state.current_chat

if current_chat:
    st.write(f"Chat: {current_chat}")
    for message in st.session_state.all_chats[current_chat]:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your message here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.all_chats[current_chat].append({'role': 'user', 'content': prompt})

        try:
            response = llm.invoke(prompt)
            st.chat_message('assistant').markdown(response)
            st.session_state.all_chats[current_chat].append({'role': 'assistant', 'content': response})
        except Exception as e:
            st.error("Failed to get a response from the model. Please try again later.")
            st.session_state.all_chats[current_chat].append({'role': 'assistant', 'content': 'Error in response.'})
else:
    st.write("No chat selected. Start a new chat or select an existing one from the sidebar.")
