import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import faiss
import numpy as np
import pickle

# Load the FAISS index
@st.cache(allow_output_mutation=True)
def load_faiss_index():
    try:
        return faiss.read_index("database/pdf_sections_index.faiss")
    except FileNotFoundError:
        st.error("FAISS index file not found. Please ensure 'pdf_sections_index.faiss' exists.")
        st.stop()

# Load the embedding model
@st.cache(allow_output_mutation=True)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load sections data
@st.cache(allow_output_mutation=True)
def load_sections_data():
    try:
        with open('database/pdf_sections_data.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Sections data file not found. Please ensure 'pdf_sections_data.pkl' exists.")
        st.stop()

# Initialize resources
index = load_faiss_index()
model = load_embedding_model()
sections_data = load_sections_data()

def search_faiss(query, k=3):
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'distance': dist,
            'content': sections_data[idx]['content'],
            'metadata': sections_data[idx]['metadata']
        })
    
    return results

prompt_template = """
You are an AI assistant specialized in dietary guidelines. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

@st.cache(allow_output_mutation=True)
def load_llm():
    return Ollama(model="llama3")  

llm = load_llm()
chain = LLMChain(llm=llm, prompt=prompt)

def answer_question(query):
    search_results = search_faiss(query)
    context = "\n\n".join([result['content'] for result in search_results])
    response = chain.run(context=context, question=query)
    return response, context

# Streamlit UI
st.title("Dietary Guidelines Q&A")

query = st.text_input("Enter your question about dietary guidelines:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Searching and generating answer..."):
            answer, context = answer_question(query)
            st.subheader("Answer:")
            st.write(answer)
            with st.expander("Show Context"):
                st.write(context)
    else:
        st.warning("Please enter a question.")