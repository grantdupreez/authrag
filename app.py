import os
import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
#import torch
from pinecone import Pinecone, ServerlessSpec

class RAGApplication:
    def __init__(self):
        # Initialize components
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load environment variables
        self.ANTHROPIC_API_KEY = st.secrets["auth_key"]
        self.MODEL = st.secrets["ai_model"]
        self.MAX_TOKENS = st.secrets["ai_tokens"]
        self.AI_TEMP = st.secrets["ai_temp"]
        self.PINECONE_API_KEY = st.secrets["pinecone_key"]
        
        # Initialize Pinecone
        self.init_pinecone()

    def generate_embeddings(self, texts):
        """
        Generate embeddings for given texts

        :param texts: List of text strings
        :return: List of embeddings
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=False)    
        return embeddings.tolist()

    def init_pinecone(self, index_name='claude-rag-index'):
        """
        Initialize Pinecone vector database
        
        :param index_name: Name of the Pinecone index
        """
        # Initialize Pinecone client
        pc = Pinecone(api_key=self.PINECONE_API_KEY)
     
        # Check if index exists
        existing_indexes = pc.list_indexes()
        
        # Create index if it doesn't exist
        if not any(index.name == index_name for index in existing_indexes):
            pc.create_index(
                name=index_name, 
                dimension=384,  # Matches all-MiniLM-L6-v2 model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='eu-west-1'
                )
            )
        
        # Initialize the index
        self.index = pc.Index(index_name)

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

    def initialize_documents(self):
        """
        Load and index sample documents if not already indexed
        """
        try:
            with open('data/sample_document.txt', 'r') as f:
                documents = f.read().split('\n\n')
        except FileNotFoundError:
            st.error("Sample document file not found. Please create 'data/sample_document.txt'")
            return []
    
        # Generate embeddings
        embeddings = self.generate_embeddings(documents)
    
        # Upsert to Pinecone
        self.upsert_documents(documents, embeddings)
        
        return documents

    def get_claude_response(self, query, context):
        """
        Get response from Claude with retrieval-augmented context
        
        :param query: User's query
        :param context: Retrieved context documents
        :return: Claude's response
        """
        
        if not self.ANTHROPIC_API_KEY:
            st.error("Anthropic API key is missing!")
            return "Error: API key not configured"
        try:
            client = anthropic.Anthropic(api_key=self.ANTHROPIC_API_KEY)

            
            # Construct prompt with context
            full_prompt = f"""
            Context Documents:
            {chr(10).join(context)}
    
            Human Query: {query}
    
            Based on the context documents, please provide a comprehensive and accurate response.
            """
        
            # Generate response
            response = client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
        
            return response.content[0].text
        except Exception as e:
            st.error(f"Error generating Claude response: {str(e)}")
            return f"Error: {str(e)}"

def main():
    st.title('Claude RAG Application')
    
    # Create RAG Application instance
    rag_app = RAGApplication()
    
    # Initialize documents on first run
    if 'initialized' not in st.session_state:
        documents = rag_app.initialize_documents()
        st.session_state['initialized'] = True
    
    # Query input
    query = st.text_input('Enter your query:')
    
    if query:
        # Generate query embedding
        query_embedding = rag_app.generate_embeddings(query)[0]
        
        # Retrieve context
        context_docs = rag_app.retrieve_similar_documents(query_embedding)
        
        # Get Claude response
        response = rag_app.get_claude_response(query, context_docs)
              
        # Display response
        st.subheader('Response')
        st.write(response)
        
        # Show retrieved context documents
        st.subheader('Retrieved Context')
        for i, doc in enumerate(context_docs, 1):
            st.text(f"Context {i}: {doc}")

if __name__ == '__main__':
    main()
