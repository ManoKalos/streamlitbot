# Ce chatbot utilise du RAG pour être spécialisé dans l'histoire de l'empire ottoman.
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.header('Chatbot Ottoman 🇹🇷')

# Model selection
option = st.selectbox(
     'Modèle :',
     ('gpt-3.5-turbo','gpt-3.5-turbo-instruct','gpt-3.5-turbo-1106','gpt-3.5-turbo-0125','gpt-4o'))


st.subheader('Slider')

max_tokens = st.slider('max tokens ?', 0, 1000)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_knowledge_base():
    """Load and process the Ottoman Empire knowledge base"""
    try:
        # Verify PDF file exists
        pdf_path = "Ottoman_Empire.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"{pdf_path} not found in current directory")
            
        # Load PDF document
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Check if documents loaded properly
        if not pages or len(pages) == 0:
            raise ValueError("PDF appears empty or could not be parsed")
            
        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(pages)
        
        # Check if splits are valid
        if not splits:
            raise ValueError("No text chunks created - check PDF content")
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        st.stop()

# Load knowledge base
retriever = load_knowledge_base()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about the Ottoman Empire!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Your question about the Ottoman Empire:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Retrieve relevant context
    try:
        context_docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in context_docs])
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        context = "No context available"
    
    # Prepare system message with context
    system_message = {
        "role": "system",
        "content": f"""You are a historian specializing in the Ottoman Empire. Answer the question using ONLY the following context. 
        If the answer isn't in the context, say you don't know. Keep answers concise and factual.
        
        Context:
        {context}"""
    }
    
    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=option,
                messages=[system_message, {"role": "user", "content": prompt}],
                stream=True,
                max_tokens=max_tokens,
                temperature=0.3
            )
            response = st.write_stream(stream)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
            st.error(response)
        
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})