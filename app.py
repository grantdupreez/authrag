import os
import uuid
import hmac
import streamlit as st
import anthropic
from typing import List, Optional, Union
from contextlib import contextmanager
from time import time

from langchain_community.document_loaders import (
    TextLoader, 
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Constants
class RAGConfig:
    """Configuration constants for the RAG application."""
    MAX_DOCS_LIMIT = 10
    MAX_COLLECTION_COUNT = 20
    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 1000
    SUPPORTED_FILE_TYPES = {
        "application/pdf": PyPDFLoader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
        "text/plain": TextLoader,
        "text/markdown": TextLoader
    }

class RAGChatApp:
    def __init__(self):
        """Initialize the RAG Chat Application."""
        self._validate_secrets()
        self._initialize_session_state()
        self._setup_authentication()

    def _validate_secrets(self):
        """Validate required secrets are present."""
        required_secrets = ["passwords", "openai_key"]
        missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]
        
        if missing_secrets:
            st.error(f"Missing required secrets: {', '.join(missing_secrets)}")
            st.stop()

    def _initialize_session_state(self):
        """Initialize session state variables with default values."""
        default_states = {
            "session_id": str(uuid.uuid4()),
            "rag_sources": [],
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I assist you today?"}
            ],
            "use_rag": True  # Default to using RAG
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _setup_authentication(self):
        """Set up login authentication."""
        if not self._check_password():
            st.stop()

    def _check_password(self) -> bool:
        """Authenticate user credentials."""
        def login_form():
            with st.form("Credentials"):
                st.text_input("Username", key="username")
                st.text_input("Password", type="password", key="password")
                st.form_submit_button("Log in", on_click=password_entered)

        def password_entered():
            username = st.session_state["username"]
            password = st.session_state["password"]
            
            if (username in st.secrets["passwords"] and 
                hmac.compare_digest(
                    password,
                    st.secrets.passwords[username]
                )):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False

        return st.session_state.get("password_correct", False) or login_form() is not None

    @contextmanager
    def _temp_file_context(self, file):
        """Context manager for temporary file handling."""
        os.makedirs("source_files", exist_ok=True)
        file_path = f"./source_files/{file.name}"
        
        try:
            with open(file_path, "wb") as f:
                f.write(file.read())
            yield file_path
        finally:
            os.remove(file_path)

    def _get_document_loader(self, file_path: str, file_type: str):
        """Select appropriate document loader based on file type."""
        loader_class = RAGConfig.SUPPORTED_FILE_TYPES.get(file_type)
        return loader_class(file_path) if loader_class else None

    def _load_documents(self, docs_to_load):
        """Load and process documents for vector database."""
        loaded_docs = []
        for doc_file in docs_to_load:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < RAGConfig.MAX_DOCS_LIMIT:
                    try:
                        with self._temp_file_context(doc_file) as file_path:
                            loader = self._get_document_loader(file_path, doc_file.type)
                            if loader:
                                loaded_docs.extend(loader.load())
                                st.session_state.rag_sources.append(doc_file.name)
                            else:
                                st.warning(f"Unsupported document type: {doc_file.type}")

                    except Exception as e:
                        st.toast(f"Error loading {doc_file.name}: {e}", icon="⚠️")
                else:
                    st.error(f"Maximum documents reached ({RAGConfig.MAX_DOCS_LIMIT}).")
        
        return loaded_docs

    def load_documents(self):
        """Process uploaded documents and add to vector database."""
        if not hasattr(st.session_state, 'rag_docs') or not st.session_state.rag_docs:
            return

        docs = self._load_documents(st.session_state.rag_docs)
        if docs:
            self._split_and_load_docs(docs)
            st.toast(f"Documents loaded: {[doc.name for doc in st.session_state.rag_docs]}", icon="✅")

    def _split_and_load_docs(self, docs: List[Document]):
        """Split documents and add to vector database."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        )
        document_chunks = text_splitter.split_documents(docs)

        if not hasattr(st.session_state, 'vector_db'):
            st.session_state.vector_db = self._initialize_vector_db(document_chunks)
        else:
            st.session_state.vector_db.add_documents(document_chunks)

    def _initialize_vector_db(self, docs: List[Document]) -> Chroma:
        """Initialize vector database with embedding."""
        embedding: Embeddings = OpenAIEmbeddings(api_key=st.secrets["openai_key"])

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=f"{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}",
        )

        self._manage_vector_db_collections(vector_db)
        return vector_db

    def _manage_vector_db_collections(self, vector_db: Chroma):
        """Manage number of vector database collections."""
        chroma_client = vector_db._client
        collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
        
        while len(collection_names) > RAGConfig.MAX_COLLECTION_COUNT:
            chroma_client.delete_collection(collection_names[0])
            collection_names.pop(0)

    def run(self):
        """Main application runner."""
        with st.sidebar:
            # Chat settings toggles
            st.toggle("Use RAG", key="use_rag", value=True)
            st.button("Clear Chat", on_click=self._clear_chat, type="primary")
            
            st.header("RAG Sources:")
            st.file_uploader(
                "📄 Upload document", 
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True,
                on_change=self.load_documents,
                key="rag_docs",
            )

            st.text_input(
                "🌐 Add URL", 
                placeholder="https://example.com",
                on_change=self._load_url,
                key="rag_url",
            )

        self._display_messages()
        
        if prompt := st.chat_input("Your message"):
            self._process_user_message(prompt)

    def _clear_chat(self):
        """Clear chat messages and reset session state."""
        st.session_state.messages.clear()
        if hasattr(st.session_state, 'vector_db'):
            del st.session_state.vector_db
        st.session_state.rag_sources.clear()

    def _load_url(self):
        """Load URL content into vector database."""
        url = st.session_state.rag_url
        if url and url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < RAGConfig.MAX_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    st.session_state.rag_sources.append(url)

                    if docs:
                        self._split_and_load_docs(docs)
                        st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")
            else:
                st.error(f"Maximum number of documents reached ({RAGConfig.MAX_DOCS_LIMIT}).")

    def _display_messages(self):
        """Display chat messages."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _process_user_message(self, prompt: str):
        """Process user message and generate response."""
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Note: The missing stream_llm_response and llm_stream functions need to be implemented
            with st.spinner("Generating response..."):
                if not st.session_state.use_rag:
                    response = stream_llm_response(llm_stream, prompt)
                else:
                    response = stream_llm_rag_response(llm_stream, prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

llm_stream = ChatAnthropic(
    api_key=anthropic_api_key,
    model=st.session_state.model.split("/")[-1],
    temperature=0.3,
    streaming=True,
)



def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})



def main():
    st.set_page_config(page_title="RAG Chat App", page_icon="🤖")
    st.title("RAG Chat Application")
    app = RAGChatApp()
    app.run()

if __name__ == "__main__":
    main()
