"""
Memory module for Spectral-JAX.

This module contains memory-related components:
- rag.py: Retrieval-Augmented Generation (VectorStore)
- Future: Episodic memory, working memory, etc.
"""

from src.memory.rag import VectorStore, Document, rag_augmented_prompt

__all__ = [
    'VectorStore',
    'Document',
    'rag_augmented_prompt',
]
