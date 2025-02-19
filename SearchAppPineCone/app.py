import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"] if "embedding" in response else None
    except Exception:
        return None

def search_similar(query_text, top_k=3):
    """Search for similar embeddings in Pinecone."""
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return []
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    unique_results = []
    seen_texts = set()
    if "matches" in results:
        for match in results["matches"]:
            text = match['metadata']['text']
            score = match['score']
            if text not in seen_texts:
                unique_results.append((text, score))
                seen_texts.add(text)
    
    return unique_results

# Streamlit UI
st.title("🔍 AI-Powered Search App")
st.write("Enter a query and find similar content stored in Pinecone.")

# Search Input
query = st.text_input("Enter your search query:", "")

if st.button("Search"):
    if query:
        results = search_similar(query)
        
        if results:
            st.subheader("Results:")
            for text, score in results:
                st.write(f"**{text}** (Score: {score:.4f})")
        else:
            st.write("❌ No similar content found.")
