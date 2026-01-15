"""
OpenAI Embeddings Client for generating text embeddings.

Uses text-embedding-3-small model (1536 dimensions) as specified in PRD.
"""

import asyncio
import os
from typing import Optional
from openai import AsyncOpenAI


class EmbeddingsClient:
    """
    Client for generating text embeddings using OpenAI's API.
    """
    
    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    MAX_BATCH_SIZE = 2048
    MAX_TOKENS_PER_REQUEST = 8191
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embeddings client.
        
        Args:
            api_key: Optional OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self._lock = asyncio.Lock()
    
    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            1536-dimensional embedding vector
            
        Raises:
            ValueError: If text is empty
            Exception: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        text = self._preprocess_text(text)
        
        try:
            response = await self.client.embeddings.create(
                model=self.MODEL,
                input=text,
                dimensions=self.DIMENSIONS
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    async def get_embeddings_batch(
        self,
        texts: list[str],
        show_progress: bool = False
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to print progress updates
            
        Returns:
            List of embedding vectors (same order as input texts)
            
        Note:
            Empty or whitespace-only texts will have None in their position.
        """
        if not texts:
            return []
        
        processed_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(self._preprocess_text(text))
                valid_indices.append(i)
        
        if not processed_texts:
            return [None] * len(texts)
        
        embeddings_map = {}
        
        batches = self._create_batches(processed_texts, valid_indices)
        
        for batch_num, (batch_texts, batch_indices) in enumerate(batches):
            if show_progress:
                print(f"Processing embedding batch {batch_num + 1}/{len(batches)} ({len(batch_texts)} texts)")
            
            try:
                response = await self.client.embeddings.create(
                    model=self.MODEL,
                    input=batch_texts,
                    dimensions=self.DIMENSIONS
                )
                
                for j, embedding_data in enumerate(response.data):
                    original_index = batch_indices[j]
                    embeddings_map[original_index] = embedding_data.embedding
                    
            except Exception as e:
                print(f"Error in batch {batch_num + 1}: {e}")
                for idx in batch_indices:
                    embeddings_map[idx] = None
        
        result = []
        for i in range(len(texts)):
            if i in embeddings_map:
                result.append(embeddings_map[i])
            else:
                result.append(None)
        
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation.
        
        - Truncate to approximate token limit
        - Remove excessive whitespace
        - Basic normalization
        """
        text = ' '.join(text.split())
        
        approx_chars = self.MAX_TOKENS_PER_REQUEST * 4
        if len(text) > approx_chars:
            text = text[:approx_chars]
        
        return text
    
    def _create_batches(
        self,
        texts: list[str],
        indices: list[int]
    ) -> list[tuple[list[str], list[int]]]:
        """
        Split texts into batches respecting API limits.
        
        Args:
            texts: Preprocessed texts
            indices: Original indices of texts
            
        Returns:
            List of (batch_texts, batch_indices) tuples
        """
        batches = []
        current_batch_texts = []
        current_batch_indices = []
        current_batch_chars = 0
        
        for text, idx in zip(texts, indices):
            text_chars = len(text)
            
            if (len(current_batch_texts) >= self.MAX_BATCH_SIZE or 
                (current_batch_chars + text_chars > self.MAX_TOKENS_PER_REQUEST * 4 * 10)):
                if current_batch_texts:
                    batches.append((current_batch_texts, current_batch_indices))
                current_batch_texts = []
                current_batch_indices = []
                current_batch_chars = 0
            
            current_batch_texts.append(text)
            current_batch_indices.append(idx)
            current_batch_chars += text_chars
        
        if current_batch_texts:
            batches.append((current_batch_texts, current_batch_indices))
        
        return batches
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        embeddings = await self.get_embeddings_batch([text1, text2])
        
        if embeddings[0] is None or embeddings[1] is None:
            return 0.0
        
        return self._cosine_similarity(embeddings[0], embeddings[1])
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Find the most similar texts to a query from a list of candidates.
        
        Args:
            query: The query text
            candidates: List of candidate texts to compare against
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity descending
        """
        all_texts = [query] + candidates
        embeddings = await self.get_embeddings_batch(all_texts)
        
        query_embedding = embeddings[0]
        if query_embedding is None:
            return []
        
        similarities = []
        for i, emb in enumerate(embeddings[1:]):
            if emb is not None:
                sim = self._cosine_similarity(query_embedding, emb)
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
