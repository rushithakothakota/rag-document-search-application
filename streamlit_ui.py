import streamlit as st
import os
from rag_pipeline import load_rag_pipeline  

st.set_page_config(page_title="AI Document Search")

st.title("📄 Pure AI Document RAG")
st.write("Upload your files and ask questions about them.")

USER_UPLOADS_DIR = "uploaded_docs"
os.makedirs(USER_UPLOADS_DIR, exist_ok=True)

with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if st.button("Clear All Documents"):
        for filename in os.listdir(USER_UPLOADS_DIR):
            os.remove(os.path.join(USER_UPLOADS_DIR, filename))
        st.cache_resource.clear()
        st.success("Knowledge base cleared!")
        st.rerun()

if uploaded_files:
    new_file_added = False
    for file in uploaded_files:
        file_path = os.path.join(USER_UPLOADS_DIR, file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            new_file_added = True
    
    if new_file_added:
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def get_rag_chain(signature):
    return load_rag_pipeline(user_uploads_dir=USER_UPLOADS_DIR)

file_sig = tuple(sorted(os.listdir(USER_UPLOADS_DIR)))
rag_chain = get_rag_chain(file_sig)

if rag_chain is None:
    st.info("No documents uploaded yet. Please upload files to start asking questions.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready. Upload some documents and ask me anything about them!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if rag_chain is None:
            st.warning("Please upload documents first.")
        else:
            with st.spinner("Searching documents..."):
                try:
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")