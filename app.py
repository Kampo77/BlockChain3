import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os

# --- CONFIG ---
st.set_page_config(page_title="AI Assistant - Kazakhstan Constitution")
st.title("📘 AI Assistant: Constitution of Kazakhstan")
st.markdown(
    """
    This application allows you to upload the Constitution of Kazakhstan in PDF format and ask questions about its content.
    The AI Assistant will provide answers based on the uploaded document.
    
    Made by Yerassyl Salimgerey, Ansar Shangilov, and Dias Trudkhanov
    """
)
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key.")
    st.stop()

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader("📎 Upload one or more PDF documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        all_documents.extend(docs)

    # Split documents
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_documents)

    # Embeddings & Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="./chroma_store")

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key),
        retriever=vectordb.as_retriever()
    )

    # --- QUERY ---
    query = st.text_input("💬 Ask a question about the documents:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.success(answer)

    # Clean temp files
    for uploaded_file in uploaded_files:
        try:
            os.remove(tmp_file_path)
        except Exception:
            pass
else:
    st.info("Please upload PDF files containing the Constitution or related content.")
