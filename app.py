import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

def set_custom_theme():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .css-1d391kg {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #2e54a5;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1e3c7b;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .uploadedFile {
            background-color: #e8f0fe;
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
        }
        .stFileUploader div>div{
                color:#333 !important;
        }
        .stFileUploader label{
                color: #666 !important;
        }
        .stTextInput>div>div>input {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #e0e0e0;
            color: #333;
        }
        h1, h2, h3 {
            color: #1e3c7b;
        }
        .app-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 30px;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .success-message {
            color: #2e54a5 !important;
            background-color: #e8f0fe;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #2e54a5;
        }
        .stAlert {
            background-color: #e8f0fe;
            color: #2e54a5;
        }
        .upload-section {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .chat-message {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #2e54a5;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

# Existing functions remain the same
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_table_from_text(text):
    # Define the table headers (you may need to adjust this based on your PDF structure)
    headers = ["Column1", "Column2", "Column3"]  # Adjust based on your table structure
    
    # Split the text into rows based on a line break or other delimiters
    rows = text.split("\n")  # Adjust based on your table's row delimiter
    
    # Initialize a list to hold the table data
    table_data = []
    
    # Loop through each row, splitting by the delimiter (space, tab, etc.)
    for row in rows:
        # Assuming space or tab separates columns in each row
        columns = row.split()  # Adjust the delimiter if needed
        if len(columns) == len(headers):  # Ensure it matches the expected number of columns
            table_data.append(columns)
    
    # Create a pandas DataFrame from the table data
    df = pd.DataFrame(table_data, columns=headers)
    return df

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    with st.container():
        with st.spinner("Generating response..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True)
            
            # Check if the response contains a table and display it
            if "table" in response["output_text"].lower():
                # Assuming the table is within the response text, parse it
                table_df = extract_table_from_text(response["output_text"])
                st.dataframe(table_df)  # Display the table using Streamlit's dataframe component
            else:
                st.markdown(f"""
                    <div class="chat-message">
                        <p style='color: #1e3c7b; font-weight: bold;'>Response:</p>
                        <p>{response["output_text"]}</p>
                    </div>
                """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    set_custom_theme()
    
    # Create two columns with custom widths
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.markdown("""
            <div class="app-header">
                <img src="https://static.vecteezy.com/system/resources/previews/010/927/083/original/chatbot-icon-on-white-background-online-support-service-bot-sign-chat-bot-sign-for-support-service-concept-flat-style-vector.jpg" style="width: 48px; height: 48px;">
                <h1 style='color:rgb(10, 31, 77); margin: 0;'>
                    Interactive PDF Chat Assistant
                </h1>
            </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input(
            "Ask your question",
            placeholder="Enter your question here...",
            key="question_input"
        )
        
        if user_question:
            user_input(user_question)
    
    with col2:
        st.markdown("""
            <div class="upload-section">
                <h3 style='color:rgb(11, 27, 63); margin-bottom: 20px;'>
                    <img src="https://img.icons8.com/color/48/000000/folder-invoices--v1.png" style="width: 24px; height: 24px; vertical-align: middle;">
                    Upload Documents
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "",
            accept_multiple_files=True,
            type=['pdf']
        )

        
        if st.button("Process Documents", key="process_btn"):
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.markdown("""
                    <div class="success-message">
                        ✅ Documents processed successfully!
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
