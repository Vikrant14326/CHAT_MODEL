import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging
import json
import base64
from datetime import datetime
import sqlite3

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Generative AI API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logging.error("Google API key not found. Make sure .env file is set up correctly.")
genai.configure(api_key=api_key)

# Initialize a global list to store query history
query_history = []

# Connect to the SQLite database
conn = sqlite3.connect('documents.db')
c = conn.cursor()

# Create the documents table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS documents
             (id INTEGER PRIMARY KEY, document_type TEXT, document_content TEXT)''')

# Create the query_history table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS query_history
             (id INTEGER PRIMARY KEY, user_id TEXT, query TEXT, response TEXT, timestamp TEXT)''')

conn.commit()

def get_document_text(document, document_type):
    """Extract text from different document types."""
    if document_type == 'pdf':
        pdf_reader = PdfReader(document)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif document_type == 'docx':
        return docx2txt.process(document)
    elif document_type == 'txt':
        return document.read()
    else:
        return ""

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate embeddings and create FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logging.info("FAISS index successfully created and saved.")

def get_conversational_chain():
    """Load conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", 
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, user_id):
    """Process user input and generate response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if the FAISS index file exists before attempting to load it
    if not os.path.exists("faiss_index/index.faiss"):
        logging.error("FAISS index file not found. Ensure that the index is created and saved properly.")
        return "Error: FAISS index file not found."

    # Load FAISS index with the necessary flag
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Load conversational chain
    chain = get_conversational_chain()

    # Generate response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_text = response["output_text"]

    # Store query and response in the history
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    query_history.append((user_id, user_question, response_text, current_time))

    # Store query and response in the database
    c.execute("INSERT INTO query_history (user_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
              (user_id, user_question, response_text, current_time))
    conn.commit()
    
    return response_text

def display_query_history(user_id):
    """Display the history of queries and responses for a specific user."""
    st.sidebar.subheader("Query History")
    c.execute("SELECT query, response, timestamp FROM query_history WHERE user_id = ?", (user_id,))
    history = c.fetchall()
    for query, response, timestamp in history:
        st.sidebar.write(f"**Query:** {query}")
        st.sidebar.write(f"**Response:** {response}")
        st.sidebar.write(f"**Timestamp:** {timestamp}")
        st.sidebar.write("---")

def download_query_history(user_id):
    """Allow users to download their query history as a JSON file."""
    c.execute("SELECT query, response, timestamp FROM query_history WHERE user_id = ?", (user_id,))
    history = c.fetchall()
    history_json = json.dumps([{"query": query, "response": response, "timestamp": timestamp} for query, response, timestamp in history], indent=4)
    b64 = base64.b64encode(history_json.encode()).decode()  # Encode the history as base64
    href = f'<a href="data:file/json;base64,{b64}" download="query_history.json">Download Query History</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

def main():
    """Main Streamlit application function."""
    st.set_page_config("Chat with Documents")
    st.header("📄📄 Chat with Documents 📄📄")

    user_id = st.text_input("Enter your user ID:")

    user_question = st.text_input("Ask a Question from the Documents")

    if user_question and user_id:
        response = user_input(user_question, user_id)
        st.write("Reply: ", response)

    with st.sidebar:
        st.title("Menu:")
        document_type = st.selectbox("Select Document Type", ["pdf", "docx", "txt"])
        document = st.file_uploader(f"Upload your {document_type.upper()} Documents", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    if document:
                        for doc in document:
                            doc_text = get_document_text(doc, document_type)
                            text_chunks = get_text_chunks(doc_text)
                            get_vector_store(text_chunks)
                            c.execute("INSERT INTO documents (document_type, document_content) VALUES (?, ?)",
                                      (document_type, doc_text))
                        conn.commit()
                        st.success("Documents processed and stored in the database.")
                    else:
                        st.error("Please upload documents before processing.")
                except Exception as e:
                    logging.error("Error processing documents: %s", e)
                    st.error(f"An error occurred: {e}")

        # Display the query history in the sidebar
        display_query_history(user_id)
        
        # Add download button for query history
        download_query_history(user_id)

if __name__ == "__main__":
    main()