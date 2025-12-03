"""
RAG (Retrieval-Augmented Generation) Implementation.

This module provides a simple vector store for storing and retrieving
text chunks based on similarity. Useful for providing external context
to the model during generation.

Key Components:
1. VectorStore: In-memory storage of text embeddings
2. Embedding: Simple character n-gram based embeddings (can be replaced with neural embeddings)
3. Retrieval: Cosine similarity based search

Future Extensions:
- FAISS/Annoy for large-scale retrieval
- Neural embeddings (BERT, sentence-transformers)
- Persistent storage (SQLite, Redis)
- Multi-modal retrieval (text + code + images)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import hashlib
import json


def simple_embedding(text: str, n: int = 3, dim: int = 256) -> np.ndarray:
    """
    Create a simple character n-gram based embedding.

    This is a placeholder for more sophisticated embeddings.
    In production, use pre-trained embeddings like BERT or sentence-transformers.

    Args:
        text: Input text
        n: N-gram size
        dim: Embedding dimension

    Returns:
        Embedding vector of shape (dim,)
    """
    # Character n-grams
    text = text.lower()
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]

    # Hash-based embedding (simple but effective)
    embedding = np.zeros(dim, dtype=np.float32)

    for ngram in ngrams:
        # Hash to get index
        hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        idx = hash_val % dim
        embedding[idx] += 1.0

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: Vector 1
        b: Vector 2

    Returns:
        Cosine similarity in [0, 1] (normalized from [-1, 1])
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine similarity: [-1, 1]
    similarity = dot_product / (norm_a * norm_b)

    # Normalize to [0, 1]
    return (similarity + 1.0) / 2.0


class Document:
    """
    A document in the knowledge base.

    Attributes:
        text: Document text
        metadata: Optional metadata (source, timestamp, etc.)
        embedding: Precomputed embedding
        doc_id: Unique document ID
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None,
        doc_id: Optional[str] = None
    ):
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding
        self.doc_id = doc_id or self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique document ID from text hash."""
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def compute_embedding(self, embed_fn=simple_embedding) -> np.ndarray:
        """
        Compute and cache embedding for this document.

        Args:
            embed_fn: Embedding function

        Returns:
            Document embedding
        """
        if self.embedding is None:
            self.embedding = embed_fn(self.text)
        return self.embedding

    def to_dict(self) -> Dict:
        """Serialize document to dictionary."""
        return {
            'text': self.text,
            'metadata': self.metadata,
            'doc_id': self.doc_id,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Document':
        """Deserialize document from dictionary."""
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        return Document(
            text=data['text'],
            metadata=data.get('metadata'),
            embedding=embedding,
            doc_id=data.get('doc_id')
        )


class VectorStore:
    """
    In-memory vector store for document retrieval.

    Stores documents with their embeddings and supports
    similarity-based retrieval.

    Example:
        >>> store = VectorStore()
        >>> store.add_documents([
        ...     "Python is a programming language.",
        ...     "JAX is a numerical computing library.",
        ...     "Neural networks learn from data."
        ... ])
        >>> results = store.search("What is JAX?", top_k=2)
        >>> print(results[0].text)
        "JAX is a numerical computing library."
    """

    def __init__(self, embed_fn=simple_embedding):
        """
        Initialize vector store.

        Args:
            embed_fn: Function to compute embeddings
        """
        self.documents: List[Document] = []
        self.embed_fn = embed_fn

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a single document to the store.

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            Document ID
        """
        doc = Document(text=text, metadata=metadata)
        doc.compute_embedding(self.embed_fn)
        self.documents.append(doc)
        return doc.doc_id

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add multiple documents to the store.

        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts

        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [None] * len(texts)

        doc_ids = []
        for text, metadata in zip(texts, metadatas):
            doc_id = self.add_document(text, metadata)
            doc_ids.append(doc_id)

        return doc_ids

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Document]:
        """
        Search for documents similar to query.

        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold [0, 1]

        Returns:
            List of top-k most similar documents
        """
        if not self.documents:
            return []

        # Compute query embedding
        query_embedding = self.embed_fn(query)

        # Compute similarities
        similarities = []
        for doc in self.documents:
            if doc.embedding is None:
                doc.compute_embedding(self.embed_fn)

            sim = cosine_similarity(query_embedding, doc.embedding)
            similarities.append((sim, doc))

        # Filter by minimum similarity
        similarities = [(sim, doc) for sim, doc in similarities if sim >= min_similarity]

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top-k documents
        top_docs = [doc for _, doc in similarities[:top_k]]

        return top_docs

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents with similarity scores.

        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.documents:
            return []

        # Compute query embedding
        query_embedding = self.embed_fn(query)

        # Compute similarities
        similarities = []
        for doc in self.documents:
            if doc.embedding is None:
                doc.compute_embedding(self.embed_fn)

            sim = cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, sim))

        # Filter by minimum similarity
        similarities = [(doc, sim) for doc, sim in similarities if sim >= min_similarity]

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return similarities[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted, False if not found
        """
        for i, doc in enumerate(self.documents):
            if doc.doc_id == doc_id:
                del self.documents[i]
                return True
        return False

    def clear(self):
        """Clear all documents from the store."""
        self.documents = []

    def size(self) -> int:
        """Return the number of documents in the store."""
        return len(self.documents)

    def save(self, filepath: str):
        """
        Save the vector store to a JSON file.

        Args:
            filepath: Path to save file
        """
        data = {
            'documents': [doc.to_dict() for doc in self.documents]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load the vector store from a JSON file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = [Document.from_dict(doc_data) for doc_data in data['documents']]


def rag_augmented_prompt(query: str, store: VectorStore, top_k: int = 3) -> str:
    """
    Create a RAG-augmented prompt by retrieving relevant context.

    Args:
        query: User query
        store: Vector store with knowledge base
        top_k: Number of context documents to retrieve

    Returns:
        Augmented prompt with context
    """
    # Retrieve relevant documents
    docs = store.search(query, top_k=top_k)

    if not docs:
        return query

    # Build context
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[Kaynak {i+1}] {doc.text}")

    context = "\n".join(context_parts)

    # Build augmented prompt
    augmented = f"""BAĞLAM (Context):
{context}

SORU (Question):
{query}

CEVAP (Answer):
"""

    return augmented


if __name__ == "__main__":
    # Test the vector store
    print("Testing RAG VectorStore...\n")

    # Create store
    store = VectorStore()

    # Add some knowledge
    knowledge_base = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "JAX is a numerical computing library for high-performance machine learning research.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
        "Transformers are a type of neural network architecture that use self-attention mechanisms.",
        "GPT (Generative Pre-trained Transformer) is a language model based on transformer architecture.",
        "JAX provides automatic differentiation and can compile Python code to run on GPUs and TPUs.",
        "Flax is a neural network library built on top of JAX for flexibility and performance.",
    ]

    print(f"Adding {len(knowledge_base)} documents to knowledge base...")
    store.add_documents(knowledge_base)
    print(f"✓ Added {store.size()} documents\n")

    # Test queries
    test_queries = [
        "What is JAX?",
        "Tell me about neural networks",
        "How does Python work?",
        "Explain transformers",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = store.search_with_scores(query, top_k=3)

        print(f"\nTop {len(results)} Results:")
        for i, (doc, score) in enumerate(results):
            print(f"\n{i+1}. [Similarity: {score:.3f}]")
            print(f"   {doc.text}")

    # Test RAG-augmented prompt
    print(f"\n\n{'='*60}")
    print("RAG-Augmented Prompt Example")
    print(f"{'='*60}")

    query = "What is JAX used for?"
    augmented_prompt = rag_augmented_prompt(query, store, top_k=2)
    print(augmented_prompt)

    # Test save/load
    print(f"\n{'='*60}")
    print("Testing Save/Load")
    print(f"{'='*60}")

    save_path = "test_vectorstore.json"
    store.save(save_path)
    print(f"✓ Saved to {save_path}")

    # Load into new store
    new_store = VectorStore()
    new_store.load(save_path)
    print(f"✓ Loaded {new_store.size()} documents")

    # Verify
    results = new_store.search("JAX", top_k=1)
    print(f"✓ Verification: Retrieved '{results[0].text[:50]}...'")

    print("\n[OK] RAG VectorStore test passed!")
