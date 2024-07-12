# NutriNudge
This project implements a question-answering system for dietary guidelines using a combination of document embedding, vector search, and language model-based generation.

![App Screenshot](image/app_screenshot.png)

## Setup

1. Clone the repository:
```
git clone https://github.com/chakraborty-arnab/NutriNudge.git
```
2. Move into the Folder:
```
cd NutriNudge
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Download and install Ollama from [Ollama](https://ollama.com/)

5. Pull the Llama3 model using Ollama:
```
ollama pull llama3
```

## Creating the Vector Database

1. Ensure you have the PDF file `Dietary_Guidelines_for_Americans_2020-2025.pdf` in the `data/` directory.

2. Run the script to create the vector database:
```
python create_vectordb.py
```
This will create two files in the `database/` directory:
- `pdf_sections_index.faiss`: The FAISS index file
- `pdf_sections_data.pkl`: The pickle file containing section data

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```
The application will be accessible in your web browser at `http://localhost:8501`.

## Usage

1. Enter your question about dietary guidelines in the text input field.
2. Click the "Get Answer" button.
3. The system will search for relevant information and generate an answer.
4. The answer will be displayed, and you can expand the "Show Context" section to see the relevant text used to generate the answer.

## Files

- `create_vectordb.py`: Script to create the vector database from the PDF document.
- `app.py`: The main Streamlit application for the Q&A system.
- `data/`: Directory containing the source PDF.
- `database/`: Directory where the vector database files are stored.

## Note

Ensure that you have sufficient disk space and computational resources to run the vector database creation and the Streamlit application. The performance may vary depending on your hardware capabilities.
