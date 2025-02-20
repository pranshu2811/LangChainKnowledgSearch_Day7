import streamlit as st
import os
import difflib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from pinecone import Pinecone
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Allowed file types and their corresponding loaders
LOADERS = {"pdf": PyPDFLoader, "txt": TextLoader, "html": BSHTMLLoader}

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Predefined FAQ responses
FAQ_RESPONSES = {
    "what is your name?": "I am an AI-powered FAQ bot designed to assist with your queries.",
    "how does this bot work?": "I provide answers using predefined FAQs and document-based knowledge. If needed, I generate AI responses.",
    "what is ai?": "Artificial Intelligence (AI) enables machines to learn, reason, and make decisions like humans.",
    "who created you?": "I was built using Python, LangChain, Pinecone, and Google's Gemini AI.",
}

# Set for tracking unique knowledge entries
new_knowledge = set()

def fuzzy_match_faq(query):
    """Finds the closest FAQ match based on similarity."""
    query = query.strip().lower()
    possible_questions = list(FAQ_RESPONSES.keys())

    closest_match = difflib.get_close_matches(query, possible_questions, n=1, cutoff=0.6)
    if closest_match:
        return FAQ_RESPONSES[closest_match[0]]
    return None

def load_and_store(file_path):
    """Loads, splits, and stores document embeddings in Pinecone."""
    ext = file_path.split(".")[-1].lower()
    loader_cls = LOADERS.get(ext)
    if not loader_cls:
        return st.error("Unsupported file format!")

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

def extract_relevant_snippet(text, query, num_sentences=2):
    """Extracts the most relevant snippet of text based on the query."""
    sentences = text.split(". ")
    for sentence in sentences:
        if query.lower() in sentence.lower():
            start_idx = max(0, sentences.index(sentence))
            end_idx = min(len(sentences), start_idx + num_sentences)
            return ". ".join(sentences[start_idx:end_idx]) + "."
    return ". ".join(sentences[:num_sentences]) + "."  # Return first few sentences as fallback

def search(query):
    """Checks FAQ first, then searches the knowledge base dynamically, and returns a single relevant result."""
    
    # First, try to get an FAQ response with fuzzy matching
    faq_response = fuzzy_match_faq(query)
    if faq_response:
        return faq_response, 1.0  # Return with max confidence

    # Otherwise, perform embedding-based search
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    results = index.query(vector=embedding, top_k=3, include_metadata=True).get("matches", [])

    if results and len(results) > 0:
        # Get the top match (first result)
        top_match = results[0]
        extracted_text = extract_relevant_snippet(top_match['metadata']['text'], query)
        score = top_match['score']

        # Add to new knowledge if it's a new finding
        new_knowledge.add((extracted_text, score))

        return extracted_text, score

    # If no results found, generate AI-based response
    response = genai.generate_text(model="gemini-pro", prompt=query)
    ai_response = response.text
    new_knowledge.add((ai_response, 1.0))
    
    return ai_response, 1.0

    """Checks FAQ first, then searches the knowledge base dynamically, and returns a single relevant result."""
    
    # First, try to get an FAQ response with fuzzy matching
    faq_response = fuzzy_match_faq(query)
    if faq_response:
        return faq_response, 1.0  # Return with max confidence

    # Otherwise, perform embedding-based search
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    results = index.query(vector=embedding, top_k=3, include_metadata=True).get("matches", [])

    if results:
        # Get the top result
        top_match = results["matches"][0]
        extracted_text = extract_relevant_snippet(top_match['metadata']['text'], query)
        score = top_match['score']

        # Add to new knowledge if it's a new finding
        new_knowledge.add((extracted_text, score))

        return extracted_text, score

    # If no results found, generate AI-based response
    response = genai.generate_text(model="gemini-pro", prompt=query)
    ai_response = response.text
    new_knowledge.add((ai_response, 1.0))
    
    return ai_response, 1.0

# Streamlit UI
st.title("AI-Powered FAQ Bot")
st.write("Upload documents to dynamically update the knowledge base.")

# File upload handling
uploaded_files = st.file_uploader("Upload documents (PDF, TXT, HTML)", type=["pdf", "txt", "html"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_and_store(file_path)

# Query input
query = st.text_input("Ask a Question:")

if st.button("Search") and query:
    result_text, result_score = search(query)

    st.write("### **Search Result:**")
    st.write(f"- {result_text} (Score: {result_score:.4f})")

    # Display newly added knowledge
    if new_knowledge:
        st.write("\n### **New Knowledge Added:**")
        for knowledge in list(new_knowledge)[-3:]:  # Show only the last 3
            st.write(f"- {knowledge[0]} (Score: {knowledge[1]:.4f})")
