# flake8: noqa: F401

import asyncio
import os
from pathlib import Path
from typing import List
import logging

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various file formats"""

    def __init__(self):
        self.supported_extensions = [".txt", ".md", ".py", ".json", ".csv"]

    async def load_documents(self, documents_path: Path) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []

        if not documents_path.exists():
            logger.warning(f"Documents path does not exist: {documents_path}")
            return documents

        for file_path in documents_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                try:
                    doc = await self._load_single_document(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading document {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    async def _load_single_document(self, file_path: Path) -> Document:
        """Load a single document file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get file stats
            stat = file_path.stat()

            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "extension": file_path.suffix,
                "file_size": stat.st_size,
                "modified_time": stat.st_mtime,
            }

            return Document(page_content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
