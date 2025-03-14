# RAG-Chatbot Using FAISS

A general-purpose conversational AI application that allows users to ask questions about any PDF document. Built with Python, Streamlit, AWS Bedrock, and FAISS vector database for efficient document retrieval.

---

## **Overview**

The **PDF Chatbot** utilizes **Retrieval-Augmented Generation (RAG)** architecture to provide context-aware and accurate responses to user queries about PDF documents. This application allows users to upload any PDF document, processes it to create searchable vector embeddings, and facilitates a natural language conversation about the document’s content.

---

## **Key Features**

- **Upload Any PDF**: Process and analyze any PDF document easily.
- **Conversational Interface**: Ask questions in natural language about the document content.
- **Source Attribution**: View the specific parts of the document that were used to generate responses.
- **Persistent Knowledge Base**: Save and reuse document embeddings for faster access to frequently queried documents.
- **Responsive UI**: A clean, user-friendly interface with real-time progress tracking.

---

## **Technical Architecture**

- **Frontend**: Streamlit web application.
- **Document Processing**: LangChain PyPDFLoader for efficient PDF parsing.
- **Embedding Generation**: AWS Bedrock Titan Embeddings for creating document embeddings.
- **Vector Database**: FAISS (Facebook AI Similarity Search) for fast and efficient retrieval of relevant document chunks.
- **Language Model**: AWS Bedrock Claude 3.5 Sonnet for high-quality natural language understanding and generation.

---

## **Implementation Guide**

### **Prerequisites**

- Python 3.8+.
- AWS account with access to Bedrock services.
- AWS credentials properly configured.

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

### **Step 2: Set Up Environment**

```bash
#Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install required packages**

```bash
pip install -r requirements.txt
```

### **Step 3: Configure AWS Credentials**

Ensure your AWS credentials are properly configured using one of the following methods:

- **AWS CLI:** Run aws configure to set up your AWS credentials.
- **Environment Variables:** Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
- **Credentials File:** Use the ~/.aws/credentials file to store your credentials.

Ensure that your AWS user or role has permissions to access Bedrock services.

### **Step 4: Run the Application**

```bash
streamlit run app.py
```

This will start the application and open it in your default web browser (typically at http://localhost:8501).

### **Step 5: Using the Application**

- **Upload a PDF:** Use the sidebar uploader to select any PDF document
- **Process the Document:** Click "Process Document" to create embeddings
- **Ask Questions:** Use the chat interface to ask questions about the document content
- **View Sources:** Expand the "View sources" section under answers to see where information was retrieved from

***

## Technical Details

- Document chunking is performed using RecursiveCharacterTextSplitter with 100-token chunks and 10-token overlap
- The application uses FAISS to store document embeddings locally for faster repeated access
- Each query retrieves the top 3 most relevant document chunks for context
- AWS Bedrock Claude 3.5 Sonnet model provides high-quality natural language understanding and response generation

***

## Performance Considerations

- Initial document processing time depends on document size and complexity
- Local FAISS index improves performance for repeated queries
- AWS Bedrock API calls may incur usage costs according to your AWS account pricing

***

## Future Enhancements

- Support for additional document formats (DOCX, TXT, etc.)
- Enhanced document summarization capabilities
- Multi-document knowledge bases
- Multi-language inputs
- User feedback mechanisms for response quality improvement

