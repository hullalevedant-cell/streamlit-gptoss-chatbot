import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Your LLM
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="your-model")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)

parser = StrOutputParser()


st.set_page_config(
    page_title="GPT OSS",
    page_icon="💬",
    layout="wide",
)

# -------------------------
# Visual design only
# -------------------------
st.markdown(
    """
    <style>
    :root { color-scheme: dark; --page:#212121; --surface:#2b2b2b; --text:#ececec; --muted:#a1a1a1; --border:rgba(255,255,255,.10); }
    .stApp { background:var(--page); color:var(--text); }
    [data-testid="stHeader"] { background:transparent; height:0; }
    [data-testid="stToolbar"], #MainMenu, footer, [data-testid="stSidebar"] { visibility:hidden; }
    .block-container { max-width:1080px; padding:2rem 2rem 7rem; }
    h1 { display:none; }
    .app-heading { text-align:center; margin:.25rem 0 2.2rem; }
    .app-heading h2 { color:var(--text); font-size:1.55rem; font-weight:600; letter-spacing:-.025em; margin:0 0 .25rem; }
    .app-heading p { color:var(--muted); font-size:.88rem; margin:0; }
    [data-testid="stChatMessage"] { gap:.85rem; margin:.7rem auto; padding:.5rem .9rem; max-width:850px; border:0; border-radius:0; background:transparent; box-shadow:none; }
    [data-testid="stChatMessage"][data-message-author-role="user"] { flex-direction:row-reverse; }
    [data-testid="stChatMessage"][data-message-author-role="assistant"] { background:transparent; }
    [data-testid="stChatMessageContent"] { color:var(--text); line-height:1.75; }
    [data-testid="stChatMessageContent"] p { margin-bottom:.65rem; }
    [data-testid="stChatMessageContent"] p:last-child { margin-bottom:0; }
    [data-testid="stChatMessage"][data-message-author-role="user"] [data-testid="stChatMessageContent"] {
        width:fit-content; max-width:min(78%,680px); margin-left:auto; padding:.75rem 1rem;
        border:1px solid var(--border); border-radius:20px; background:var(--surface);
    }
    [data-testid="stChatMessage"][data-message-author-role="user"] [data-testid="chatAvatarIcon-user"] { display:none; }
    [data-testid="stChatMessage"][data-message-author-role="assistant"] [data-testid="chatAvatarIcon-assistant"] { color:#dedede; background:#303030; border-radius:50%; }
    [data-testid="stChatMessageContent"] pre { border:1px solid var(--border); border-radius:12px; background:#181818; }
    [data-testid="stChatMessageContent"] code:not(pre code) { color:#e6e6e6; background:#333; border-radius:5px; padding:.12rem .32rem; }
    [data-testid="stChatInput"] { max-width:850px; margin:0 auto; padding-bottom:1rem; }
    [data-testid="stChatInput"] > div {
        border:1px solid rgba(255,255,255,.13) !important; border-radius:26px !important;
        background:#2b2b2b !important; box-shadow:0 4px 18px rgba(0,0,0,.18);
        transition:border-color .16s ease, box-shadow .16s ease;
    }
    [data-testid="stChatInput"] > div:focus-within { border-color:rgba(255,255,255,.28) !important; box-shadow:0 0 0 2px rgba(255,255,255,.04), 0 4px 18px rgba(0,0,0,.18); }
    [data-testid="stChatInput"] textarea { color:var(--text) !important; caret-color:#ddd; }
    [data-testid="stChatInput"] textarea::placeholder { color:#929292 !important; }
    [data-testid="stChatInput"] button { color:#171717 !important; background:#e4e4e4 !important; border:0 !important; border-radius:50% !important; }
    [data-testid="stChatInput"] button:hover { background:#fff !important; }
    @media (max-width:640px) {
        .block-container { padding:1.5rem .6rem 6rem; }
        .app-heading { margin-bottom:1.6rem; }
        .app-heading h2 { font-size:1.35rem; }
        [data-testid="stChatMessage"] { padding:.4rem .25rem; }
        [data-testid="stChatMessage"][data-message-author-role="user"] [data-testid="stChatMessageContent"] { max-width:88%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session state
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("system", "You are a helpful chatbot. Be concise and accurate.")
    ]


# -------------------------
# UI
st.markdown(
    '<div class="app-heading"><h2>GPT OSS</h2><p>AI Assistant</p></div>',
    unsafe_allow_html=True,
)
# -------------------------

st.title("💬 AI Chat Assistant")

st.markdown(
    '<p style="display:none;">Ask me anything.</p>',
    unsafe_allow_html=True,
)


# Display previous messages
for role, message in st.session_state.chat_history:

    if role == "system":
        continue

    with st.chat_message(role):
        st.markdown(message)


# -------------------------
# User input
# -------------------------

user_input = st.chat_input("Type a message...")


if user_input:

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.chat_history.append(
        ("user", user_input)
    )

    # Build prompt from entire conversation
    prompt = ChatPromptTemplate.from_messages(
        st.session_state.chat_history
    )

    # Build chain
    chain = prompt | llm | parser

    # Get response
    response = chain.invoke({})

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant response
    st.session_state.chat_history.append(
        ("assistant", response)
    )
