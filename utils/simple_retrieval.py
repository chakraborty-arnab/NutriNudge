from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the FAISS index
index = faiss.read_index("database/pdf_sections_index.faiss")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_faiss(query, k=3):
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)
    return distances, indices

# Example usage
query = "What are the main dietary guidelines for protein intake?"
distances, indices = search_faiss(query)

print(f"Query: {query}")
print(f"Distances: {distances}")
print(f"Indices: {indices}")