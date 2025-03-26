import os
import streamlit as st
import pinecone

# Load environment variables
ANTHROPIC_API_KEY=st.secrets["auth_key"]
MODEL=st.secrets["ai_model"]
MAX_TOKENS=st.secrets["ai_tokens"]
AI_TEMP=st.secrets["ai_temp"]
PINECONE_API_KEY=st.secrets["pincecone_key"]
PINECONE_ENVIRONMENT=st.secrets["pinecone_env"]

class PineconeRetriever:
    def __init__(self, index_name='claude-rag-index'):
        """
        Initialize Pinecone vector database
        
        :param index_name: Name of the Pinecone index
        """
        load_dotenv()
        
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT')
        )
        
        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name, 
                dimension=384,  # Matches all-MiniLM-L6-v2 model
                metric='cosine'
            )
        
        self.index = pinecone.Index(index_name)
    
    def upsert_documents(self, documents, embeddings):
        """
        Upsert documents and their embeddings into Pinecone
        
        :param documents: List of document texts
        :param embeddings: Corresponding list of embeddings
        """
        # Create vectors with unique IDs
        vectors = [
            (str(i), embedding, {'text': doc}) 
            for i, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]
        
        # Upsert to Pinecone
        self.index.upsert(vectors)
    
    def retrieve_similar_documents(self, query_embedding, top_k=3):
        """
        Retrieve most similar documents
        
        :param query_embedding: Embedding of the query
        :param top_k: Number of documents to retrieve
        :return: List of retrieved documents
        """
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        # Extract and return documents
        return [
            match.metadata['text'] 
            for match in results['matches']
        ]
