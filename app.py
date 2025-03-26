import os
import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
import torch
import pinecone


# Load environment variables
ANTHROPIC_API_KEY=st.secrets["auth_key"]
MODEL=st.secrets["ai_model"]
MAX_TOKENS=st.secrets["ai_tokens"]
AI_TEMP=st.secrets["ai_temp"]
PINECONE_API_KEY=st.secrets["pinecone_key"]
PINECONE_ENVIRONMENT=st.secrets["pinecone_env"]

# Initialize components


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

def init_Pinecone(self, index_name='claude-rag-index'):
    """
    Initialize Pinecone vector database
    
    :param index_name: Name of the Pinecone index
    """

    # Load environment variables
    PINECONE_API_KEY=st.secrets["pinecone_key"]
    PINECONE_ENVIRONMENT=st.secrets["pinecone_env"]
        
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


def initialize_documents():
    """
    Load and index sample documents if not already indexed
    """
    with open('data/sample_document.txt', 'r') as f:
        documents = f.read().split('\n\n')
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(documents)
    
    # Upsert to Pinecone
    pinecone_retriever.upsert_documents(documents, embeddings)

def get_claude_response(query, context):
    """
    Get response from Claude with retrieval-augmented context
    
    :param query: User's query
    :param context: Retrieved context documents
    :return: Claude's response
    """
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    # Construct prompt with context
    full_prompt = f"""
    Context Documents:
    {chr(10).join(context)}

    Human Query: {query}

    Based on the context documents, please provide a comprehensive and accurate response.
    """
    
    # Generate response
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    
    return response.content[0].text

def main():
    st.title('Claude RAG Application')
    
    # Initialize documents on first run
    if 'initialized' not in st.session_state:
        initialize_documents()
        st.session_state['initialized'] = True
    
    # Query input
    query = st.text_input('Enter your query:')
    
    if query:
        # Generate query embedding
        query_embedding = embedding_generator.generate_embeddings(query)[0]
        
        # Retrieve context
        context_docs = pinecone_retriever.retrieve_similar_documents(query_embedding)
        
        # Get Claude response
        response = get_claude_response(query, context_docs)
        
        # Display response
        st.subheader('Response')
        st.write(response)
        
        # Show retrieved context documents
        st.subheader('Retrieved Context')
        for i, doc in enumerate(context_docs, 1):
            st.text(f"Context {i}: {doc}")

if __name__ == '__main__':
    main()
