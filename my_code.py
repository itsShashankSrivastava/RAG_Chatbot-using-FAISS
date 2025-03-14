import json
import os
import boto3
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot using FAISS",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f7ff;
    }
    .chat-message .avatar {
        width: 50px;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .chat-message .avatar img {
        max-width: 50px;
        max-height: 50px;
        border-radius: 50%;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .feedback-buttons {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-top: 5px;
    }
    .sources-container {
        margin-top: 10px;
        border-left: 3px solid #ccc;
        padding-left: 10px;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize AWS Bedrock connection using boto3
@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

bedrock_client = get_bedrock_client()

def create_embeddings(documents, embeddings_model):
    """
    Create embeddings for the given documents
    """
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=100,
            chunk_overlap=10
        )
        split_texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = BedrockEmbeddings(
            client=bedrock_client, 
            model_id=embeddings_model
        )
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(split_texts, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def document_index(indexpath="faiss_index", pdf_file_path=None):
    """
    Loads or creates FAISS index with progress tracking
    """
    # Create progress placeholder
    progress_container = st.empty()
    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        embedding_model = "amazon.titan-embed-text-v2:0"
        
        # Check if we need to rebuild the index with a new PDF
        force_rebuild = pdf_file_path is not None
        
        # Check if local FAISS index exists and we're not forcing a rebuild
        if os.path.exists(os.path.join(indexpath, "index.faiss")) and not force_rebuild:
            status_text.info("Loading existing knowledge base...")
            progress_bar.progress(30)
            time.sleep(0.5)  # Small delay for better UX
            
            embeddings = BedrockEmbeddings(
                client=bedrock_client, 
                model_id=embedding_model
            )
            progress_bar.progress(60)
            time.sleep(0.5)  # Small delay for better UX
            
            vectorstore = FAISS.load_local(indexpath, embeddings, allow_dangerous_deserialization=True)
            
            progress_bar.progress(100)
            status_text.success("Knowledge base loaded successfully!")
            time.sleep(1)  # Let the success message display for a moment
            return vectorstore

        # If no existing index or forcing rebuild, create new one
        status_text.info("Preparing new knowledge base...")
        progress_bar.progress(10)
        time.sleep(0.5)  # Small delay for better UX

        # Ensure index directory exists
        os.makedirs(indexpath, exist_ok=True)

        # If no PDF specified, use default
        if pdf_file_path is None:
            # Check if we have a default PDF set in session state
            if "default_pdf" in st.session_state and st.session_state.default_pdf:
                pdf_file_path = st.session_state.default_pdf
            else:
                status_text.error("No PDF file specified. Please upload a document.")
                return None
        
        # Validate PDF exists
        if not os.path.exists(pdf_file_path):
            status_text.error(f"PDF file not found: {pdf_file_path}")
            return None

        status_text.info(f"Processing document: {os.path.basename(pdf_file_path)}...")
        progress_bar.progress(30)
        time.sleep(0.5)  # Small delay for better UX
        
        data_load = PyPDFLoader(pdf_file_path)
        documents = data_load.load_and_split()
        
        status_text.info("Creating embeddings from document content...")
        progress_bar.progress(50)
        time.sleep(0.5)  # Small delay for better UX

        # Create embeddings
        vectorstore = create_embeddings(documents, embedding_model)
        
        if vectorstore is None:
            status_text.error("Failed to create embeddings")
            return None

        # Save the index locally
        status_text.info("Saving knowledge base...")
        progress_bar.progress(80)
        time.sleep(0.5)  # Small delay for better UX
        
        vectorstore.save_local(indexpath)

        progress_bar.progress(100)
        status_text.success("Knowledge base created successfully!")
        time.sleep(1)  # Let the success message display for a moment
        
        return vectorstore

    except Exception as e:
        status_text.error(f"An error occurred: {str(e)}")
        return None
    finally:
        # Clear the progress container after completion
        time.sleep(1)
        progress_container.empty()

def create_llm():
    """Creates a connection to Bedrock LLM."""
    def query_bedrock(prompt):
        # Format the prompt for Claude
        body_data = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        try:
            response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body_data),
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            return response_body['content'][0]['text']
        except Exception as e:
            st.error(f"Error connecting to LLM: {str(e)}")
            return "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."
    return query_bedrock

def generate_rag_response(index, question):
    """Retrieves best match from FAISS and queries LLM."""
    try:
        # Get the k most relevant document chunks
        k = 3 
        docs = index.similarity_search_with_score(question, k=k)
        
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(docs):
            context_parts.append(f"[Document {i+1}]:\n{doc.page_content}")
            # Track source information
            source_info = getattr(doc, 'metadata', {})
            page_num = source_info.get('page', 'unknown page')
            sources.append(f"Document {i+1}: Page {page_num} (relevance score: {score:.2f})")
        
        context = "\n\n".join(context_parts)
        
        # Craft a detailed prompt for the LLM
        prompt = f"""You are an assistant chatbot that provides precise and relevant information based solely on 
        the data provided. Answer questions using the available information, and ensure your responses are clear,
        professional, and friendly. If the required information is unavailable, politely inform the user without making assumptions.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        # Get response from LLM
        llm_response = create_llm()(prompt)
        
        return {
            "answer": llm_response,
            "sources": sources
        }
    except Exception as e:
        st.error(f"Error in RAG response: {str(e)}")
        return {
            "answer": "I encountered an error while processing your question. Please try again.",
            "sources": []
        }

def add_message(role, content, sources=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = {"role": role, "content": content, "timestamp": timestamp}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)

def display_chat_messages():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(f"{message['content']}")
                st.caption(f"Asked at {message['timestamp']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"{message['content']}")
                st.caption(f"Answered at {message['timestamp']}")
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("View sources"):
                        for source in message["sources"]:
                            st.write(source)

def reset_chat_history():
    st.session_state.messages = []

# Streamlit app main logic
def main():
    # Sidebar for app configuration
    with st.sidebar:
        st.markdown(
            """
            <style>
                .circular-img {
                    border-radius: 50%;
                    width: 100px;
                    height: 100px;
                    object-fit: cover;
                }
            </style>
            <img class="circular-img" src="https://png.pngtree.com/png-vector/20220707/ourmid/pngtree-chatbot-robot-concept-chat-bot-png-image_5632381.png" />
            """, 
            unsafe_allow_html=True
        )
        st.title("PDF Chatbot")
        st.write("Your AI-powered guide to any PDF document")
        
        st.divider()
        
        # Upload new policy document
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file to use with the index
            bytes_data = uploaded_file.getvalue()
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(bytes_data)
            
            # Store the default PDF path in session state
            st.session_state.default_pdf = temp_file_path
            
            if st.button("Process Document"):
                with st.spinner("Processing your document..."):
                    # Reset the vectorstore with the new document
                    st.session_state.vectorstore = document_index(pdf_file_path=temp_file_path)
                    # Clear chat history for the new document
                    reset_chat_history()
                    st.success(f"Successfully processed {uploaded_file.name}")
        
        st.divider()
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            reset_chat_history()
            st.success("Chat history cleared!")
        
        st.divider()
        
        # App information
        st.subheader("About")
        st.write("""
        This PDF Chatbot helps you find information from any PDF document.
        Upload a document and ask questions about its content.
        """)
        
        st.write("Powered by AWS Bedrock and Langchain")
        
    # Main content area
    st.title("PDF Chatbot 📚")
    
    # Check if there's a default PDF in session state
    if 'default_pdf' not in st.session_state:
        st.info("Please upload a PDF document in the sidebar to get started.")
        return
    
    # Initialize vectorstore in session state if not exists
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        with st.spinner('Initializing knowledge base... This may take a moment.'):
            st.session_state.vectorstore = document_index(pdf_file_path=st.session_state.default_pdf)

    # Check if vectorstore is successfully created
    if st.session_state.vectorstore is None:
        st.error("Failed to create or load knowledge base. Please check your PDF and configuration.")
        return
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        display_chat_messages()
    
    # Question input
    st.divider()
    question = st.chat_input("Ask a question about the document (e.g., 'What are the main topics discussed?')")

    # Response generation
    if question:
        # Add user question to chat
        add_message("user", question)
        
        # Display updated chat with user's new question
        with chat_container:
            display_chat_messages()
        
        # Generate response with a spinner
        with st.spinner('Searching document...'):
            response_data = generate_rag_response(st.session_state.vectorstore, question)
            
            # Add assistant response to chat
            add_message("assistant", response_data["answer"], response_data["sources"])
        
        # Display updated chat with new response
        with chat_container:
            display_chat_messages()

# Run the app
if __name__ == "__main__":
    main()
