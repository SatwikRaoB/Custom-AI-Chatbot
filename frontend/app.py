"""
Streamlit Frontend for Customer Support Chatbot
Reads BACKEND_URL and AUTH_TOKEN from environment variables.
Supports optional web search toggle and reindexing.
"""

import os
import requests
import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Customer Support Chatbot", layout="centered")

# --- Backend URL & Auth ---
BACKEND_URL = os.getenv("BACKEND_URL", "https://custom-ai-chatbot-54452819884.us-central1.run.app")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

st.sidebar.title("⚙️ Settings")

backend_url = st.sidebar.text_input("Backend URL", BACKEND_URL)
auth_token = st.sidebar.text_input("Auth Token", AUTH_TOKEN, type="password")

if backend_url.endswith("/"):
    backend_url = backend_url[:-1]

# Web search options
enable_web_search = st.sidebar.checkbox("Enable web search (Google CSE)", value=False)
web_results_num = st.sidebar.number_input("Search results to fetch", min_value=1, max_value=10, value=5)

# Reindex button
if st.sidebar.button("Reindex backend"):
    try:
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        res = requests.post(f"{backend_url}/reindex", headers=headers, timeout=120)
        if res.ok:
            st.sidebar.success("✅ Reindex successful")
        else:
            st.sidebar.error(f"❌ Reindex failed: {res.status_code} {res.text[:200]}")
    except Exception as e:
        st.sidebar.error(f"Reindex error: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    "Place PDF files in `data/manuals/` then click **Reindex**.\n\n"
    "Web search requires `GOOGLE_SEARCH_API_KEY` and `GOOGLE_CSE_ID` set in backend."
)

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "text": "You are chatting with the Customer Support Bot."}]

st.title("💬 Customer Support Chatbot")
st.markdown("Ask questions based on uploaded PDFs. Toggle *Enable web search* to include live results.")

# --- Chat Form ---
with st.form("ask_form"):
    question = st.text_area("Your question", value="", height=100)
    submitted = st.form_submit_button("Send")

if submitted and question.strip():
    with st.spinner("Contacting backend..."):
        payload = {
            "question": question,
            "web_search": bool(enable_web_search),
            "web_num_results": int(web_results_num),
        }
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

        try:
            res = requests.post(f"{backend_url}/ask", json=payload, headers=headers, timeout=60)
            if res.status_code != 200:
                st.error(f"Backend returned {res.status_code}: {res.text}")
            else:
                data = res.json()
                answer = data.get("answer", "No answer returned.")
                web_results = data.get("web_results")
                source_context = data.get("source_context", "")

                st.session_state.messages.append(
                    {"role": "user", "text": question, "time": datetime.now().isoformat()}
                )
                st.session_state.messages.append(
                    {
                        "role": "bot",
                        "text": answer,
                        "time": datetime.now().isoformat(),
                        "web_results": web_results,
                        "source_context": source_context,
                    }
                )
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

# --- Render Messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    elif msg["role"] == "bot":
        st.markdown(f"**Bot:** {msg['text']}")
        if msg.get("web_results"):
            with st.expander("Web search results (snippet)"):
                st.write(msg["web_results"])
        if msg.get("source_context"):
            with st.expander("Source context (excerpt)"):
                st.write(msg["source_context"])

st.markdown("---")
st.markdown("💡 Tip: Add PDFs to `data/manuals/` and click **Reindex** in the sidebar.")
