import os
from dotenv import load_dotenv
from pinecone import Pinecone

# ✅ Load environment variables
load_dotenv()

# ✅ Explicitly retrieve API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ✅ Ensure API key is loaded correctly
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("❌ Missing Pinecone API Key or Index Name. Check your .env file!")

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Delete all vectors from the index
index.delete(delete_all=True)

print("✅ All vectors have been deleted from Pinecone.")
