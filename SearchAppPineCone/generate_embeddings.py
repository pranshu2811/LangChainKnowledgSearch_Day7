import google.generativeai as genai
from config import GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text):
    """Generate text embeddings using Gemini API."""
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"] if "embedding" in response else None
    except Exception:
        return None
if __name__ == "__main__":
    sample_text = "Hello world"
    embedding = get_embedding(sample_text)
    if embedding:
        print(embedding[:10])  
