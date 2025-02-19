import os
import streamlit as st
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ Explicitly set the Google API Key
if not GEMINI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not PINECONE_ENVIRONMENT:
    raise ValueError("❌ Missing required API keys in config.py/.env!")

genai.configure(api_key=GEMINI_API_KEY)

# ✅ Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Allowed file types
LOADERS = {"pdf": PyPDFLoader, "txt": TextLoader, "html": BSHTMLLoader}

# ✅ Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Chatbot Pipeline (Gemini AI)
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
output_parser = StrOutputParser()
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide responses to user queries."),
    ("user", "Question:{question}")
])
chat_chain = chat_prompt | chat_model | output_parser

def load_and_store(file_path):
    """Loads, splits, and stores document embeddings in Pinecone."""
    ext = file_path.split(".")[-1].lower()
    loader_cls = LOADERS.get(ext)
    if not loader_cls:
        st.error("Unsupported file format!")
        return

    docs = loader_cls(file_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

    vectors = []
    for i, chunk in enumerate(chunks):
        if chunk.page_content:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=chunk.page_content,
                task_type="retrieval_query"
            )["embedding"]

            doc_id = f"doc_{os.path.basename(file_path)}_{i}"
            vectors.append({"id": doc_id, "values": embedding, "metadata": {"text": chunk.page_content}})
    
    if vectors:
        index.upsert(vectors=vectors)
        st.success(f"{os.path.basename(file_path)} has been processed and stored!")

def search_documents(query):
    """Searches the knowledge base and retrieves relevant results."""
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    results = index.query(vector=embedding, top_k=3, include_metadata=True).get("matches", [])

    if results:
        return [(match['metadata']['text'], match['score']) for match in results]
    
    return None

# ✅ Streamlit UI
st.title("AI-Powered Chatbot & FAQ Bot")
st.write("Upload documents to dynamically update the knowledge base.")

# ✅ File upload handling
uploaded_files = st.file_uploader("Upload documents (PDF, TXT, HTML)", type=["pdf", "txt", "html"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process and store embeddings
        load_and_store(file_path)

# ✅ Chatbot input
query = st.text_input("Ask the chatbot anything:")
if st.button("Chat") and query:
    # Step 1️⃣: Search in uploaded documents first
    document_results = search_documents(query)

    if document_results:
        st.write("### **Document-Based Answer:**")
        for text, score in document_results:
            st.write(f"- {text} (Score: {score:.4f})")
    else:
        # Step 2️⃣: If no relevant document found, use Gemini AI
        st.write("### **AI Chatbot Response:**")
        response = chat_chain.invoke({'question': query})
        st.write(response)
