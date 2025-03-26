import os
import streamlit as st
import anthropic

from utils.embedding import EmbeddingGenerator
from utils.retrieval import PineconeRetriever

# Load environment variables
ANTHROPIC_API_KEY=st.secrets["auth_key"]
MODEL=st.secrets["ai_model"]
MAX_TOKENS=st.secrets["ai_tokens"]
AI_TEMP=st.secrets["ai_temp"]
PINECONE_API_KEY=st.secrets["pinecone_key"]
PINECONE_ENVIRONMENT=st.secrets["pinecone_env"]

# Initialize components
embedding_generator = EmbeddingGenerator()
pinecone_retriever = PineconeRetriever()

def initialize_documents():
    """
    Load and index sample documents if not already indexed
    """
    with open('data/sample_documents.txt', 'r') as f:
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
