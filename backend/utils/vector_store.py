# flake8: noqa: F401, E501, E203

import asyncio
import numpy as np
from typing import List, Dict, Callable, Any
import logging
import json
import pickle
from pathlib import Path

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """Simple in-memory vector store for similarity search"""

    def __init__(self, storage_path: str = "data/vector_store"):
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def add_documents(
        self, documents: List[Document], embedding_function: Callable
    ):
        """Add documents to the vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store")

        # Extract text content
        texts = [doc.page_content for doc in documents]

        # Generate embeddings in batches to avoid API limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = await embedding_function(batch_texts)
            all_embeddings.extend(batch_embeddings)

            # Add delay to respect API limits
            if i + batch_size < len(texts):
                await asyncio.sleep(1)

        # Store embeddings and documents
        self.embeddings.extend(all_embeddings)
        self.documents.extend([doc.page_content for doc in documents])
        self.metadata.extend([doc.metadata for doc in documents])

        # Save to disk
        await self._save_to_disk()

        logger.info(f"Successfully added {len(documents)} documents to vector store")

    async def similarity_search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Dict]:
        """Find k most similar documents to query"""
        if not self.embeddings:
            logger.warning("No embeddings in vector store")
            return []

        # Calculate cosine similarities
        query_np = np.array(query_embedding)
        similarities = []

        for i, doc_embedding in enumerate(self.embeddings):
            doc_np = np.array(doc_embedding)
            similarity = np.dot(query_np, doc_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(doc_np)
            )
            similarities.append((similarity, i))

        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])

        results = []
        for similarity, idx in similarities[:k]:
            results.append(
                {
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(similarity),
                }
            )

        return results

    async def _save_to_disk(self):
        """Save vector store to disk"""
        try:
            data = {
                "embeddings": self.embeddings,
                "documents": self.documents,
                "metadata": self.metadata,
            }

            with open(self.storage_path / "vector_store.pkl", "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    async def load_from_disk(self):
        """Load vector store from disk"""
        try:
            store_file = self.storage_path / "vector_store.pkl"
            if store_file.exists():
                with open(store_file, "rb") as f:
                    data = pickle.load(f)

                self.embeddings = data["embeddings"]
                self.documents = data["documents"]
                self.metadata = data["metadata"]

                logger.info(f"Loaded {len(self.documents)} documents from disk")
                return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")

        return False
