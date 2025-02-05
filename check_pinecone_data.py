from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
stats = index.describe_index_stats()
print("Total embeddings stored in Pinecone:", stats["total_vector_count"])
