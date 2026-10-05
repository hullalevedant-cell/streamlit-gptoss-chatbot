import os
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="GPT OSS",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    :root {
        color-scheme: dark;

        --bg: #171717;
        --surface: #212121;
        --surface-hover: #292929;

        --text: #eeeeee;
        --muted: #8d8d8d;

        --border: rgba(255,255,255,0.10);
        --border-soft: rgba(255,255,255,0.07);

        --user-bg: #2b2b2b;
        --input-bg: #242424;
    }


    /* =====================================================
       APP
       ===================================================== */

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }

    [data-testid="stToolbar"],
    #MainMenu,
    footer,
    [data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        max-width: 950px !important;

        padding-top: 1.5rem !important;
        padding-bottom: 150px !important;
    }

    h1 {
        display: none !important;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .app-heading {
        text-align: center;

        margin-top: 0.2rem;
        margin-bottom: 2.5rem;
    }

    .app-heading h2 {
        margin: 0;

        color: #eeeeee;

        font-size: 1.35rem;
        font-weight: 600;

        letter-spacing: -0.025em;
    }

    .app-heading p {
        margin: 0.3rem 0 0;

        color: #858585;

        font-size: 0.78rem;
    }


    /* =====================================================
       CHAT MESSAGE BASE
       ===================================================== */

    [data-testid="stChatMessage"] {

        width: 100% !important;
        max-width: 850px !important;

        margin-left: auto !important;
        margin-right: auto !important;

        padding-top: 0.65rem !important;
        padding-bottom: 0.65rem !important;

        background: transparent !important;

        border: none !important;

        box-shadow: none !important;

        gap: 0.75rem !important;
    }


    /* =====================================================
       AVATARS
       ===================================================== */

    /* User avatar */

    [data-testid="stChatMessage"][data-message-author-role="user"]
    [data-testid="stChatMessageAvatar"] {

        width: 30px !important;
        height: 30px !important;

        min-width: 30px !important;

        border-radius: 50% !important;

        background: #363636 !important;

        color: #dcdcdc !important;

        border: 1px solid rgba(255,255,255,0.08) !important;
    }


    /* Assistant avatar */

    [data-testid="stChatMessage"][data-message-author-role="assistant"]
    [data-testid="stChatMessageAvatar"] {

        width: 30px !important;
        height: 30px !important;

        min-width: 30px !important;

        border-radius: 9px !important;

        background: #eeeeee !important;

        color: #171717 !important;

        border: none !important;
    }


    /* =====================================================
       MESSAGE CONTENT
       ===================================================== */

    [data-testid="stChatMessageContent"] {

        color: var(--text) !important;

        font-size: 0.95rem !important;

        line-height: 1.7 !important;

        min-width: 0 !important;
    }

    [data-testid="stChatMessageContent"] p {

        margin-top: 0 !important;

        margin-bottom: 0.75rem !important;
    }

    [data-testid="stChatMessageContent"] p:last-child {

        margin-bottom: 0 !important;
    }


    /* =====================================================
       USER MESSAGE
       ===================================================== */

    [data-testid="stChatMessage"][data-message-author-role="user"] {

        justify-content: flex-end !important;

        flex-direction: row-reverse !important;
    }

    [data-testid="stChatMessage"][data-message-author-role="user"]
    [data-testid="stChatMessageContent"] {

        flex: 0 1 auto !important;

        width: auto !important;

        max-width: 650px !important;

        padding: 0.7rem 1rem !important;

        background: var(--user-bg) !important;

        border: 1px solid var(--border-soft) !important;

        border-radius: 17px !important;

        color: #eeeeee !important;

        line-height: 1.55 !important;
    }


    /* =====================================================
       ASSISTANT MESSAGE
       ===================================================== */

    [data-testid="stChatMessage"][data-message-author-role="assistant"] {

        justify-content: flex-start !important;

        flex-direction: row !important;
    }

    [data-testid="stChatMessage"][data-message-author-role="assistant"]
    [data-testid="stChatMessageContent"] {

        max-width: 780px !important;

        padding: 0 !important;

        background: transparent !important;

        border: none !important;
    }


    /* =====================================================
       MARKDOWN HEADINGS
       ===================================================== */

    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3 {

        color: #f0f0f0 !important;

        letter-spacing: -0.02em !important;

        margin-top: 1.15rem !important;

        margin-bottom: 0.6rem !important;
    }

    [data-testid="stChatMessageContent"] h1 {
        font-size: 1.35rem !important;
    }

    [data-testid="stChatMessageContent"] h2 {
        font-size: 1.15rem !important;
    }

    [data-testid="stChatMessageContent"] h3 {
        font-size: 1rem !important;
    }


    /* =====================================================
       LISTS
       ===================================================== */

    [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessageContent"] ol {

        padding-left: 1.35rem !important;

        margin-top: 0.4rem !important;

        margin-bottom: 0.8rem !important;
    }

    [data-testid="stChatMessageContent"] li {

        margin-bottom: 0.35rem !important;
    }


    /* =====================================================
       INLINE CODE
       ===================================================== */

    [data-testid="stChatMessageContent"] code:not(pre code) {

        background: #303030 !important;

        color: #e6e6e6 !important;

        padding: 0.12rem 0.35rem !important;

        border-radius: 5px !important;

        font-size: 0.88em !important;
    }


    /* =====================================================
       CODE BLOCKS
       ===================================================== */

    [data-testid="stChatMessageContent"] pre {

        background: #101010 !important;

        border: 1px solid var(--border) !important;

        border-radius: 12px !important;

        padding: 1rem !important;

        margin: 1rem 0 !important;

        overflow-x: auto !important;
    }


    [data-testid="stChatMessageContent"] pre code {

        background: transparent !important;

        color: #e5e5e5 !important;

        padding: 0 !important;

        font-size: 0.86rem !important;

        line-height: 1.6 !important;
    }


    /* =====================================================
       TABLES
       ===================================================== */

    [data-testid="stChatMessageContent"] table {

        width: 100% !important;

        margin: 1rem 0 !important;

        border-collapse: separate !important;

        border-spacing: 0 !important;

        overflow: hidden !important;

        border: 1px solid var(--border) !important;

        border-radius: 10px !important;

        background: #1c1c1c !important;

        font-size: 0.86rem !important;
    }

    [data-testid="stChatMessageContent"] th {

        background: #292929 !important;

        color: #eeeeee !important;

        padding: 0.7rem 0.8rem !important;

        text-align: left !important;

        border-bottom: 1px solid var(--border) !important;
    }

    [data-testid="stChatMessageContent"] td {

        color: #d2d2d2 !important;

        padding: 0.7rem 0.8rem !important;

        border-bottom: 1px solid var(--border-soft) !important;

        vertical-align: top !important;
    }


    /* =====================================================
       INPUT BAR
       ===================================================== */

    [data-testid="stChatInput"] {

        position: fixed !important;

        left: 50% !important;

        bottom: 20px !important;

        transform: translateX(-50%) !important;

        width: min(720px, calc(100vw - 420px)) !important;

        max-width: 720px !important;

        min-width: 420px !important;

        z-index: 999999 !important;

        padding: 0 !important;

        margin: 0 !important;

        background: transparent !important;
    }


    [data-testid="stChatInput"] > div {

        min-height: 58px !important;

        background: #242424 !important;

        border: 1px solid rgba(255,255,255,0.12) !important;

        border-radius: 18px !important;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.35) !important;

        transition:
            border-color 0.15s ease,
            box-shadow 0.15s ease !important;
    }


    [data-testid="stChatInput"] > div:hover {

        border-color: rgba(255,255,255,0.18) !important;
    }


    [data-testid="stChatInput"] > div:focus-within {

        border-color: rgba(255,255,255,0.25) !important;

        box-shadow:
            0 0 0 2px rgba(255,255,255,0.035),
            0 10px 35px rgba(0,0,0,0.35) !important;
    }


    [data-testid="stChatInput"] textarea {

        color: #eeeeee !important;

        font-size: 0.95rem !important;

        caret-color: white !important;
    }


    [data-testid="stChatInput"] textarea::placeholder {

        color: #858585 !important;
    }


    /* Send button */

    [data-testid="stChatInput"] button {

        background: #eeeeee !important;

        color: #171717 !important;

        border: none !important;

        border-radius: 50% !important;
    }


    [data-testid="stChatInput"] button:hover {

        background: white !important;
    }


    /* =====================================================
       MODEL SELECTOR
       ===================================================== */

    [data-testid="stSelectbox"] {

        position: fixed !important;

        right: 28px !important;

        bottom: 20px !important;

        width: 190px !important;

        z-index: 999999 !important;

        padding: 0 !important;

        margin: 0 !important;
    }


    [data-testid="stSelectbox"] [data-baseweb="select"] > div {

        min-height: 58px !important;

        background: #242424 !important;

        border: 1px solid rgba(255,255,255,0.12) !important;

        border-radius: 18px !important;

        color: #eeeeee !important;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.25) !important;
    }


    [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {

        background: #292929 !important;

        border-color: rgba(255,255,255,0.18) !important;
    }


    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 800px) {

        .block-container {

            padding-left: 0.7rem !important;

            padding-right: 0.7rem !important;

            padding-bottom: 130px !important;
        }


        [data-testid="stChatMessage"] {

            max-width: 100% !important;
        }


        [data-testid="stChatMessage"][data-message-author-role="user"]
        [data-testid="stChatMessageContent"] {

            max-width: 78% !important;
        }


        [data-testid="stChatInput"] {

            width: calc(100vw - 30px) !important;

            min-width: 0 !important;

            bottom: 15px !important;
        }


        [data-testid="stSelectbox"] {

            display: none !important;
        }
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [
        (
            "system",
            """
            You are a helpful AI assistant.

            Be concise, accurate, and easy to understand.

            Use Markdown for formatting.
            Use proper Markdown tables when useful.
            Use Markdown code blocks for code.

            Never use HTML tags such as <br>, <div>, or <p>
            in your responses.
            """
        )
    ]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="app-heading">
        <h2>GPT OSS</h2>
        <p>AI Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL CHOICES
# =========================================================

MODEL_CHOICES = {
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "GPT OSS 20B": "openai/gpt-oss-20b",
}


parser = StrOutputParser()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for role, message in st.session_state.chat_history:

    if role == "system":
        continue

    if role == "user":

        with st.chat_message(
            "user",
            avatar=":material/person:"
        ):
            st.markdown(message)

    else:

        with st.chat_message(
            "assistant",
            avatar=":material/auto_awesome:"
        ):
            st.markdown(message)


# =========================================================
# INPUT + MODEL
# =========================================================

input_column, model_column = st.columns(
    [0.86, 0.14],
    gap="small",
    vertical_alignment="bottom",
)


with input_column:

    user_input = st.chat_input(
        "Message GPT OSS..."
    )


with model_column:

    selected_model_name = st.selectbox(
        "Model",
        options=list(MODEL_CHOICES.keys()),
        index=0,
        label_visibility="collapsed",
        key="model_selector",
    )


model_input_name = MODEL_CHOICES[selected_model_name]


# =========================================================
# GROQ
# =========================================================

llm = ChatGroq(
    model=model_input_name,
    temperature=0.1,
)


# =========================================================
# HANDLE USER INPUT
# =========================================================

if user_input:

    # Save user message
    st.session_state.chat_history.append(
        ("user", user_input)
    )


    # Build prompt
    prompt = ChatPromptTemplate.from_messages(
        st.session_state.chat_history
    )


    # Build chain
    chain = prompt | llm | parser


    # Generate response
    response = chain.invoke({})


    # Clean accidental HTML breaks
    response = (
        response
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
    )


    # Save response
    st.session_state.chat_history.append(
        ("assistant", response)
    )


    # Refresh UI
    st.rerun()