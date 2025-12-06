import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# -------------------------------------------------
# Load custom CSS
# -------------------------------------------------
with open("frontend/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="AI Document Bot", layout="wide")
st.title("📄 AI Document Bot — RAG Interface")


# -------------------------------------------------
# Helper: load docs
# -------------------------------------------------
def load_documents():
    res = requests.get(f"{API_URL}/list_documents")
    if res.status_code == 200:
        return res.json()["documents"]
    return []


# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📤 Upload / Delete Files",
    "🔍 Semantic Search",
    "🤖 RAG Chat"
])


# -------------------------------------------------
# TAB 1 — Upload & Delete
# -------------------------------------------------
with tab1:
    st.header("📤 Upload a new document")

    file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

    if file:
        with st.status("Uploading file...", expanded=False) as status:
            res = requests.post(
                f"{API_URL}/upload_file",
                files={"file": (file.name, file.getvalue())}
            )
            if res.status_code == 200:
                status.update(label=f"Uploaded: {file.name}", state="complete")
            else:
                status.update(label="Upload failed", state="error")

    # --- Documents list ---
    st.subheader("📁 Available Documents")
    docs = load_documents()
    st.write(docs if docs else "No documents uploaded.")

    # --- Delete ---
    st.subheader("🗑️ Delete a document")
    if docs:
        to_delete = st.selectbox("Select a file:", docs)

        if st.button("Delete selected file"):
            with st.status(f"Deleting {to_delete}...", expanded=False) as status:
                res = requests.delete(
                    f"{API_URL}/delete_file",
                    params={"filename": to_delete}
                )
                if res.status_code == 200:
                    status.update(label=f"Deleted: {to_delete}", state="complete")
                else:
                    status.update(label="Delete failed", state="error")
            st.rerun()

    # --- Debug chunks ---
    st.subheader("🔍 Inspect chunks")
    if docs:
        selected = st.selectbox("Select document to inspect:", docs, key="chunk_inspect")

        if st.button("Show chunks"):
            res = requests.get(f"{API_URL}/debug/chunks", params={"file": selected})
            if res.status_code == 200:
                st.json(res.json())
            else:
                st.error(res.text)


# -------------------------------------------------
# TAB 2 — Semantic Search
# -------------------------------------------------
with tab2:
    st.header("🔍 Semantic Search (No LLM)")

    query = st.text_input("Search query:")

    if st.button("Search"):
        if query.strip():
            res = requests.post(
                f"{API_URL}/search",
                json={"question": query, "top_k": 5}
            )

            if res.status_code == 200:
                results = res.json()["results"]

                for r in results:
                    st.markdown(
                        f"""
                        <div class="search-result">
                            <b>📄 {r['file']} — chunk {r['chunk']}</b><br>
                            <span>Score: {r['score']:.4f}</span><br><br>
                            {r['text']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error(res.text)
        else:
            st.warning("Query cannot be empty.")


# -------------------------------------------------
# TAB 3 — RAG Chat
# -------------------------------------------------
with tab3:
    st.header("🤖 RAG Chat — Ask a question")

    question = st.text_area("Your question:")

    if st.button("Get Answer"):
        if question.strip():
            with st.status("Generating answer...", expanded=False) as status:
                res = requests.post(
                    f"{API_URL}/rag_answer",
                    json={"question": question, "top_k": 5}
                )
                if res.status_code == 200:
                    status.update(label="Answer ready", state="complete")
                    st.success(res.json()["answer"])
                else:
                    status.update(label="Error during LLM call", state="error")
                    st.error(res.text)
        else:
            st.warning("Question cannot be empty.")
