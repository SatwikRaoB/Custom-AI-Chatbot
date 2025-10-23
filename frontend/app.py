# frontend/app.py
"""
Streamlit chat UI with optional Web Search toggle.
Save as customer-support-bot/frontend/app.py
Run: streamlit run frontend/app.py
"""
import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Customer Support Chatbot", layout="centered")

# Backend URL (safe fallback)
try:
    DEFAULT_BACKEND = st.secrets.get("backend_url", "http://127.0.0.1:8000")
except Exception:
    DEFAULT_BACKEND = "http://127.0.0.1:8000"

# Sidebar controls
st.sidebar.title("Settings")
backend_url = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND)
if backend_url.endswith("/"):
    backend_url = backend_url[:-1]

# Web search toggle and num results
enable_web_search = st.sidebar.checkbox("Enable web search (Google CSE)", value=False)
web_results_num = st.sidebar.number_input("Search results to fetch", min_value=1, max_value=10, value=5)

if st.sidebar.button("Reindex backend"):
    try:
        res = requests.post(f"{backend_url}/reindex", timeout=120)
        if res.ok:
            st.sidebar.success("Reindex successful")
        else:
            st.sidebar.error(f"Reindex failed: {res.status_code} {res.text[:200]}")
    except Exception as e:
        st.sidebar.error(f"Reindex error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Place PDF files in data/manuals/ then Reindex. Web search requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID set in backend .env.")

# Conversation state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "text": "You are chatting with the Customer Support Bot."}]

st.title("💬 Customer Support Chatbot")
st.markdown("Ask questions based on uploaded PDFs. Toggle 'Enable web search' to include live web results.")

with st.form("ask_form"):
    question = st.text_area("Your question", value="", height=100)
    submitted = st.form_submit_button("Send", key="send_button")

if submitted and question.strip():
    with st.spinner("Contacting backend..."):
        payload = {
            "question": question,
            "web_search": bool(enable_web_search),
            "web_num_results": int(web_results_num),
        }
        try:
            res = requests.post(f"{backend_url}/ask", json=payload, timeout=60)
            if res.status_code != 200:
                st.error(f"Backend returned {res.status_code}: {res.text}")
            else:
                data = res.json()
                answer = data.get("answer", "No answer returned.")
                web_results = data.get("web_results")
                source_context = data.get("source_context", "")
                st.session_state.messages.append({"role": "user", "text": question, "time": datetime.now().isoformat()})
                st.session_state.messages.append({"role": "bot", "text": answer, "time": datetime.now().isoformat(), "web_results": web_results, "source_context": source_context})
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

# render messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")
        if msg.get("web_results"):
            with st.expander("Web search results (snippet)"):
                st.write(msg.get("web_results"))
        if msg.get("source_context"):
            with st.expander("Source context (excerpt)"):
                st.write(msg.get("source_context"))

st.markdown("---")
st.markdown("Tip: Add PDFs to data/manuals/ and click Reindex in the sidebar.")
