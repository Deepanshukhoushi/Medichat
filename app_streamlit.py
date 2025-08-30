# import streamlit as st
# from app import get_answer, supabase
# from uuid import uuid4

# # ---------------------------
# # Session management
# # ---------------------------
# if "user_id" not in st.session_state:
#     st.session_state.user_id = f"guest_{uuid4()}"   # guest mode

# # ---------------------------
# # Sidebar for Auth
# # ---------------------------
# st.sidebar.title("Authentication")

# if "auth_status" not in st.session_state:
#     st.session_state.auth_status = "guest"

# if st.sidebar.button("Logout") and st.session_state.auth_status == "logged_in":
#     st.session_state.user_id = f"guest_{uuid4()}"
#     st.session_state.auth_status = "guest"
#     st.sidebar.success("Logged out!")

# with st.sidebar.expander("Login"):
#     email = st.text_input("Email", key="login_email")
#     password = st.text_input("Password", type="password", key="login_pass")
#     if st.button("Login"):
#         try:
#             res = supabase.auth.sign_in_with_password({"email": email, "password": password})
#             if getattr(res, "user", None):
#                 st.session_state.user_id = res.user.id
#                 st.session_state.auth_status = "logged_in"
#                 st.sidebar.success("Logged in!")
#             else:
#                 st.sidebar.error("Invalid credentials")
#         except Exception as e:
#             st.sidebar.error(str(e))

# with st.sidebar.expander("Signup"):
#     email = st.text_input("Email", key="signup_email")
#     password = st.text_input("Password", type="password", key="signup_pass")
#     if st.button("Signup"):
#         try:
#             supabase.auth.sign_up({"email": email, "password": password})
#             st.sidebar.success("Check email for confirmation!")
#         except Exception as e:
#             st.sidebar.error(str(e))

# # ---------------------------
# # Chat Interface
# # ---------------------------
# st.title("💬 MediChat (Streamlit UI)")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display history
# for role, msg in st.session_state.chat_history:
#     if role == "user":
#         st.chat_message("user").markdown(msg)
#     else:
#         st.chat_message("assistant").markdown(msg)

# # User input
# if user_input := st.chat_input("Ask me a medical question..."):
#     st.session_state.chat_history.append(("user", user_input))
#     st.chat_message("user").markdown(user_input)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             answer = get_answer(user_input, st.session_state.user_id)
#             st.markdown(answer)
#             st.session_state.chat_history.append(("assistant", answer))






# # app_streamlit.py
# import streamlit as st
# from uuid import uuid4
# from app import (
#     get_answer,
#     supabase,
#     ensure_conversation,
#     load_history_for_ui,
# )

# # ---------------------------
# # Session management
# # ---------------------------
# if "user_id" not in st.session_state:
#     st.session_state.user_id = f"guest_{uuid4()}"   # guest mode
# if "auth_status" not in st.session_state:
#     st.session_state.auth_status = "guest"
# if "conversation_id" not in st.session_state:
#     st.session_state.conversation_id = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # ---------------------------
# # Sidebar: Auth + Conversation
# # ---------------------------
# st.sidebar.title("Authentication")

# if st.sidebar.button("Logout") and st.session_state.auth_status == "logged_in":
#     st.session_state.user_id = f"guest_{uuid4()}"
#     st.session_state.auth_status = "guest"
#     st.session_state.conversation_id = None
#     st.session_state.chat_history = []
#     st.sidebar.success("Logged out!")

# with st.sidebar.expander("Login"):
#     email = st.text_input("Email", key="login_email")
#     password = st.text_input("Password", type="password", key="login_pass")
#     if st.button("Login"):
#         try:
#             res = supabase.auth.sign_in_with_password({"email": email, "password": password})
#             if getattr(res, "user", None):
#                 st.session_state.user_id = res.user.id
#                 st.session_state.auth_status = "logged_in"
#                 st.session_state.conversation_id = ensure_conversation(st.session_state.user_id)
#                 st.sidebar.success("Logged in!")

#                 # Load previous chat_history for this conversation
#                 st.session_state.chat_history = load_history_for_ui(st.session_state.conversation_id)
#             else:
#                 st.sidebar.error("Invalid credentials")
#         except Exception as e:
#             st.sidebar.error(str(e))

# with st.sidebar.expander("Signup"):
#     email = st.text_input("Email", key="signup_email")
#     password = st.text_input("Password", type="password", key="signup_pass")
#     if st.button("Signup"):
#         try:
#             supabase.auth.sign_up({"email": email, "password": password})
#             st.sidebar.success("Check email for confirmation!")
#         except Exception as e:
#             st.sidebar.error(str(e))

# # Optional: New chat button
# if st.sidebar.button("➕ New Chat"):
#     if st.session_state.auth_status == "logged_in":
#         try:
#             created = supabase.table("conversations").insert({
#                 "user_id": st.session_state.user_id,
#                 "title": "New Conversation"
#             }).execute()
#             st.session_state.conversation_id = created.data[0]["id"]
#             st.session_state.chat_history = []
#             st.sidebar.success("Started a new chat.")
#         except Exception as e:
#             st.sidebar.error(f"Failed to create conversation: {e}")
#     else:
#         # guest: just clear local memory
#         st.session_state.conversation_id = None
#         st.session_state.chat_history = []
#         st.sidebar.info("New guest chat started (not saved).")

# # ---------------------------
# # Chat Interface
# # ---------------------------
# st.title("💬 MediChat (Streamlit UI)")

# # Display history
# for role, msg in st.session_state.chat_history:
#     st.chat_message("user" if role == "user" else "assistant").markdown(msg)

# # User input
# if user_input := st.chat_input("Ask me a medical question..."):
#     # Ensure conversation exists for logged-in users
#     if st.session_state.auth_status == "logged_in":
#         if not st.session_state.conversation_id:
#             st.session_state.conversation_id = ensure_conversation(st.session_state.user_id)
#     # Show user message immediately
#     st.session_state.chat_history.append(("user", user_input))
#     st.chat_message("user").markdown(user_input)

#     # Get answer (passes conversation_id so memory is per-conversation)
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             answer = get_answer(
#                 user_input,
#                 st.session_state.user_id,
#                 st.session_state.conversation_id,  # None for guest; persisted for logged-in
#             )
#             st.markdown(answer)
#             st.session_state.chat_history.append(("assistant", answer))









import streamlit as st
from uuid import uuid4
from app import (
    get_answer,
    supabase,
    ensure_conversation,
    load_history_for_ui,
    get_user_conversations
)
import time
import os

# ---------------------------
# Session management
# ---------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = f"guest_{uuid4()}"   # guest mode
if "auth_status" not in st.session_state:
    st.session_state.auth_status = "guest"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_login_modal" not in st.session_state:
    st.session_state.show_login_modal = True
if "conversations" not in st.session_state:
    st.session_state.conversations = []

# ---------------------------
# Authentication Modal
# ---------------------------
if st.session_state.show_login_modal:
    with st.sidebar:
        with st.expander("Authentication", expanded=True):
            st.write("Choose how you'd like to use MediChat:")
            
            # Guest option
            if st.button("Continue as Guest"):
                st.session_state.user_id = f"guest_{uuid4()}"
                st.session_state.auth_status = "guest"
                st.session_state.conversation_id = None
                st.session_state.chat_history = []
                st.session_state.show_login_modal = False
                st.rerun()
            
            st.divider()
            
            # Google OAuth - Fixed to handle missing secrets
            if st.button("Sign in with Google"):
                try:
                    # Get the redirect URL from environment or use default
                    redirect_url = os.getenv('REDIRECT_URL', 'http://localhost:8501')
                    
                    # Get the OAuth URL
                    oauth_response = supabase.auth.sign_in_with_oauth({
                        "provider": "google",
                        "options": {
                            "redirect_to": redirect_url
                        }
                    })
                    st.markdown(f"[Complete Google Sign-in]({oauth_response.url})", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Google authentication error: {str(e)}")
            
            st.divider()
            
            # Email/password login
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        if getattr(res, "user", None):
                            st.session_state.user_id = res.user.id
                            st.session_state.auth_status = "logged_in"
                            st.session_state.conversation_id = ensure_conversation(st.session_state.user_id)
                            st.session_state.show_login_modal = False
                            
                            # Load previous chat_history for this conversation
                            st.session_state.chat_history = load_history_for_ui(st.session_state.conversation_id)
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    except Exception as e:
                        st.error(str(e))
            
            with col2:
                if st.button("Signup"):
                    try:
                        supabase.auth.sign_up({"email": email, "password": password})
                        st.success("Check email for confirmation!")
                    except Exception as e:
                        st.error(str(e))

# ... (rest of the file remains the same)

# ---------------------------
# Main App (after authentication)
# ---------------------------
if not st.session_state.show_login_modal:
    # ---------------------------
    # Sidebar: User info + Conversations
    # ---------------------------
    # ... (previous code remains the same until the sidebar section)

# ---------------------------
# Sidebar: User info + Conversations
# ---------------------------
    with st.sidebar:
        # User info
        if st.session_state.auth_status == "logged_in":
            try:
                user_info = supabase.auth.get_user()
                if user_info.user:
                    st.write(f"Logged in as: **{user_info.user.email}**")
            except:
                st.write("Logged in")
        else:
            st.write("Using as **Guest**")
            st.info("Your chat history will not be saved")
        
        # Logout button
        if st.session_state.auth_status == "logged_in" and st.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user_id = f"guest_{uuid4()}"
            st.session_state.auth_status = "guest"
            st.session_state.conversation_id = None
            st.session_state.chat_history = []
            st.session_state.show_login_modal = True
            st.rerun()
        
        # New chat button
        if st.button("➕ New Chat"):
            if st.session_state.auth_status == "logged_in":
                try:
                    created = supabase.table("conversations").insert({
                        "user_id": st.session_state.user_id,
                        "title": f"Chat {time.strftime('%Y-%m-%d %H:%M')}"
                    }).execute()
                    st.session_state.conversation_id = created.data[0]["id"]
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create conversation: {e}")
            else:
                # guest: just clear local memory
                st.session_state.conversation_id = f"guest_{uuid4()}"
                st.session_state.chat_history = []
                st.info("New guest chat started (not saved).")
                st.rerun()
        
        # Load previous conversations for logged-in users only
        if st.session_state.auth_status == "logged_in":
            st.divider()
            st.subheader("Previous Chats")
            
            try:
                # Fetch conversations
                conversations = get_user_conversations(st.session_state.user_id)
                
                for conv in conversations:
                    if st.button(conv["title"], key=f"conv_{conv['id']}"):
                        st.session_state.conversation_id = conv["id"]
                        st.session_state.chat_history = load_history_for_ui(conv["id"])
                        st.rerun()
            except Exception as e:
                st.error(f"Error loading conversations: {str(e)}")

# ... (rest of the file remains the same)

    # ---------------------------
    # Chat Interface
    # ---------------------------
    st.title("💬 MediChat")
    
    # Display conversation title if available
    if st.session_state.conversation_id and st.session_state.auth_status == "logged_in":
        try:
            res = supabase.table("conversations")\
                .select("title")\
                .eq("id", st.session_state.conversation_id)\
                .single()\
                .execute()
            if res.data:
                st.subheader(res.data["title"])
        except:
            pass
    
    # Display history
    for role, msg in st.session_state.chat_history:
        st.chat_message("user" if role == "user" else "assistant").markdown(msg)
    
    # User input
    if user_input := st.chat_input("Ask me a medical question..."):
        # Ensure conversation exists for logged-in users
        if st.session_state.auth_status == "logged_in" and not st.session_state.conversation_id:
            st.session_state.conversation_id = ensure_conversation(st.session_state.user_id)
        
        # Show user message immediately
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)
        
        # Get answer (passes conversation_id so memory is per-conversation)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    user_input,
                    st.session_state.user_id,
                    st.session_state.conversation_id,  # None for guest; persisted for logged-in
                )
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))
