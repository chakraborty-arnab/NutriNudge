import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
import faiss
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="NutriNudge", page_icon="🍎", layout="wide")

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }
    .stExpander {
        border: 1px solid #4CAF50;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

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
st.title("🍽️ NutriNudge: Dietary Guidelines Q&A")

st.markdown('<p class="big-font">Ask a question about dietary guidelines:</p>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g., What are the main food groups?")

if st.button("Get Answer"):
    if query:
        with st.spinner("Searching and generating answer..."):
            answer, context = answer_question(query)
            st.subheader("Answer:")
            st.info(answer)
            with st.expander("Show Context"):
                st.write(context)
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown("Source: [Dietary Guidelines for Americans 2020-2025](https://www.dietaryguidelines.gov/sites/default/files/2020-12/Dietary_Guidelines_for_Americans_2020-2025.pdf)")

# Add footer
st.markdown("---")

# Add GitHub and LinkedIn icons
st.markdown("""
    <style>
    .social-links {
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    .social-btn {
        display: inline-flex;
        width: 40px;
        height: 40px;
        background-color: #ffffff;
        color: #000000;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        font-size: 24px;
        transition: 0.3s;
    }
    .social-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    </style>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    
    <div class="social-links">
        <a href="https://github.com/chakraborty-arnab" target="_blank" class="social-btn">
            <i class="fab fa-github"></i>
        </a>
        <a href="https://www.linkedin.com/in/arnab-chakraborty13/" target="_blank" class="social-btn">
            <i class="fab fa-linkedin-in"></i>
        </a>
    </div>
""", unsafe_allow_html=True)

st.markdown("Developed with ❤️ using Ollama, Streamlit and LangChain")