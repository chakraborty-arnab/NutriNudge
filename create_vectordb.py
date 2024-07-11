from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load the PDF
pdf_path = "data/Dietary_Guidelines_for_Americans_2020-2025.pdf"
loader = PyPDFLoader(file_path=pdf_path)

# Load the content
documents = loader.load()

# Split the document into sections
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
sections = text_splitter.split_documents(documents)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each section
section_texts = [section.page_content for section in sections]
embeddings = model.encode(section_texts)

print(embeddings.shape)

embeddings_np = np.array(embeddings).astype('float32')

# Create a FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to the index
index.add(embeddings_np)

# Save the index to a file
faiss.write_index(index, "database/pdf_sections_index.faiss")

# When creating the index:
sections_data = [
    {
        'content': section.page_content,
        'metadata': section.metadata
    }
    for section in sections
]

# Save sections data
with open('database/pdf_sections_data.pkl', 'wb') as f:
    pickle.dump(sections_data, f)

print("Embeddings stored in FAISS index and saved to file.")
