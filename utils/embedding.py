import os
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        :param model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        
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
