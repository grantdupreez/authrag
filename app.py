import os
import uuid
import hmac
import streamlit as st
import anthropic
from typing import List, Optional

# Improved document loaders
from langchain_community.document_loaders import (
    TextLoader, 
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage, AIMessage

# Constants
DB_DOCS_LIMIT = 10
MAX_COLLECTION_COUNT = 20
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 1000

class RAGChatApp:
    def __init__(self):
        self._initialize_session_state()
        self._setup_authentication()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I assist you today?"}
            ]

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
            if (st.session_state["username"] in st.secrets["passwords"] and 
                hmac.compare_digest(
                    st.session_state["password"],
                    st.secrets.passwords[st.session_state["username"]]
                )):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False

        return st.session_state.get("password_correct", False) or login_form() is not None

    def _get_document_loader(self, file_path: str, file_type: str):
        """Select appropriate document loader based on file type."""
        loaders = {
            "application/pdf": PyPDFLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
            "text/plain": TextLoader,
            "text/markdown": TextLoader
        }
        return loaders.get(file_type, lambda x: None)(file_path)

    def load_documents(self):
        """Load documents into the vector database."""
        if not hasattr(st.session_state, 'rag_docs') or not st.session_state.rag_docs:
            return

        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    
                    try:
                        with open(file_path, "wb") as file:
                            file.write(doc_file.read())

                        loader = self._get_document_loader(file_path, doc_file.type)
                        if loader:
                            docs.extend(loader.load())
                            st.session_state.rag_sources.append(doc_file.name)
                        else:
                            st.warning(f"Unsupported document type: {doc_file.type}")

                    except Exception as e:
                        st.toast(f"Error loading {doc_file.name}: {e}", icon="⚠️")
                    
                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            self._split_and_load_docs(docs)
            st.toast(f"Documents loaded: {[doc.name for doc in st.session_state.rag_docs]}", icon="✅")

    def _split_and_load_docs(self, docs):
        """Split documents and add to vector database."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        document_chunks = text_splitter.split_documents(docs)

        if not hasattr(st.session_state, 'vector_db'):
            st.session_state.vector_db = self._initialize_vector_db(document_chunks)
        else:
            st.session_state.vector_db.add_documents(document_chunks)

    def _initialize_vector_db(self, docs):
        """Initialize vector database with embedding."""
        # Determine embedding based on available keys
        if "AZ_OPENAI_API_KEY" not in os.environ:
            embedding = OpenAIEmbeddings(api_key=st.session_state.get('openai_api_key'))
        else:
            embedding = AzureOpenAIEmbeddings(
                api_key=os.getenv("AZ_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
                model="text-embedding-3-large",
                openai_api_version="2024-02-15-preview",
            )

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=f"{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}",
        )

        # Manage collection count
        self._manage_vector_db_collections(vector_db)
        return vector_db

    def _manage_vector_db_collections(self, vector_db):
        """Manage number of vector database collections."""
        chroma_client = vector_db._client
        collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
        
        while len(collection_names) > MAX_COLLECTION_COUNT:
            chroma_client.delete_collection(collection_names[0])
            collection_names.pop(0)

    def run(self):
        """Main application runner."""
        with st.sidebar:
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
        """Clear chat messages."""
        st.session_state.messages.clear()

    def _load_url(self):
        """Load URL content into vector database."""
        # Similar implementation to load_documents, but for URLs
        pass  # Implement URL loading logic here

    def _display_messages(self):
        """Display chat messages."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _process_user_message(self, prompt):
        """Process user message and generate response."""
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Implement your LLM response generation here
            pass

def main():
    st.title("RAG Chat Application")
    app = RAGChatApp()
    app.run()

if __name__ == "__main__":
    main()
