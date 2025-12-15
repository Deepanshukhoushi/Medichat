import streamlit as st
import os
from uuid import uuid4
from app import get_answer

# Disable LangChain tracing
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_HANDLER'] = 'false'
os.environ['LANGCHAIN_TELEMETRY'] = 'false'

# Page config
st.set_page_config(
    page_title="MediChat - Medical Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = f"guest_{uuid4()}"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"guest_{uuid4()}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("🧠 MediChat")
    st.info("**Guest Mode**\nChats saved locally for this session only.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.conversation_id = f"guest_{uuid4()}"
        st.success("Chat cleared!")

    st.divider()
    st.caption("Powered by Cohere AI")

# Main interface
st.title("💬 MediChat - Medical Assistant")
st.caption("Ask me medical questions. I'll use indexed medical data when available.")

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical data..."):
            try:
                response = get_answer(
                    prompt,
                    st.session_state.user_id,
                    st.session_state.conversation_id
                )
                st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))
