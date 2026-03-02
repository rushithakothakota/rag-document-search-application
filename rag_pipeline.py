import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def load_rag_pipeline(user_uploads_dir="uploaded_docs"):
    all_documents = []

    if os.path.exists(user_uploads_dir):
        for filename in os.listdir(user_uploads_dir):
            file_path = os.path.join(user_uploads_dir, filename)
            loader = None
            ext = filename.lower()
            
            if ext.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif ext.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif ext.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")

            if loader:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["filename"] = filename
                all_documents.extend(docs)  

    if not all_documents:
                return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_db = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)

    system_prompt = (
        "You are a professional document assistant. "
        "Use ONLY the retrieved context to answer the question. "
        "If the answer is not in the context, say that: "
        "'I don’t have information in the uploaded documents.' "
        "Do not use outside knowledge."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)