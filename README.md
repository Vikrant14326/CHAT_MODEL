# Chat with Documents

https://github.com/user-attachments/assets/58662770-d6e9-47f6-bb5f-f26b352be9ee

This is a Streamlit-based application that allows you to upload PDF, DOCX, or TXT documents and then ask questions about the content of those documents. The application uses the LangChain library and the Google Generative AI API to provide natural language processing and question-answering capabilities.

## Features

- Upload multiple documents of different types (PDF, DOCX, TXT)
- Split the documents into manageable chunks and create a FAISS index for efficient retrieval
- Ask questions about the uploaded documents and receive detailed answers
- Store the query history for each user
- Allow users to download their query history as a JSON file

## Requirements

The application requires the following Python packages:

- `streamlit`
- `google-generativeai`
- `python-dotenv`
- `langchain`
- `PyPDF2`
- `chromadb`
- `faiss-cpu==1.7.1`
- `langchain_google_genai`
- `langchain_community`

You can install these packages by running the following command:
## Setup

1. Create a `.env` file in the root directory of your project and add your Google API key:
2. Run the Streamlit application:
The application will start running, and you can access it in your web browser at `http://localhost:8501`.

## Usage

1. Select the document type you want to upload (PDF, DOCX, or TXT) from the sidebar.
2. Upload your documents by clicking the "Browse files" button and selecting the files.
3. Click the "Submit & Process" button to process the documents and create the FAISS index.
4. Enter your user ID in the input field.
5. Ask a question about the uploaded documents in the "Ask a Question from the Documents" input field.
6. The application will return the answer to your question, based on the content of the documents.

The query history for the user will be displayed in the sidebar, and you can download the history as a JSON file by clicking the "Download Query History" link.

## Customization

You can customize the application by modifying the following aspects:

- **Prompt Template**: You can update the prompt template used for the question-answering chain in the `get_conversational_chain()` function.
- **Logging**: You can adjust the logging configuration in the `logging.basicConfig()` call to change the log level or output format.
- **Database**: You can modify the SQLite database schema or switch to a different database system if needed.

## License

This project is licensed under the [MIT License](LICENSE).

