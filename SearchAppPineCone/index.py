import pinecone
from config import PINECONE_API_KEY

pinecone.init(api_key=PINECONE_API_KEY)
print(pinecone.list_indexes())