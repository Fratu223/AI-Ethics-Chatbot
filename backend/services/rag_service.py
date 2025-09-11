# flake8: noqa: F401, E501, W291, E722

import asyncio
import time
import json
import hashlib
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import random

import openai
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from models.schemas import QueryResponse, RetrievedChunk, DocumentResponse
from utils.document_loader import DocumentLoader
from utils.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        # Initialize API clients with proper error handling
        self.openai_available = False
        self.cohere_available = False

        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if (
                openai_key
                and openai_key.strip()
                and openai_key != "your_openai_api_key_here"
            ):
                self.openai_client = openai.AsyncOpenAI(
                    api_key=openai_key, timeout=60.0, max_retries=2
                )
                self.openai_available = True
                logger.info("OpenAI client initialized")
            else:
                logger.warning(
                    "OpenAI API key not found - will use fallback text search"
                )

            # Make Cohere optional
            cohere_key = os.getenv("COHERE_API_KEY")
            if (
                cohere_key
                and cohere_key.strip()
                and cohere_key != "your_cohere_api_key_here"
            ):
                self.cohere_client = cohere.AsyncClient(api_key=cohere_key)
                self.cohere_available = True
                logger.info("Cohere client initialized")
            else:
                logger.warning("Cohere API key not found - reranking disabled")

        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
            self.openai_available = False
            self.cohere_available = False

        # Initialize components
        self.document_loader = DocumentLoader()
        self.vector_store = VectorStore()

        # Use smaller chunks for free tier (less embedding cost)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Even smaller
            chunk_overlap=100,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        self.documents_metadata = []
        self.is_initialized = False

        # Rate limiting for free tier
        self.last_embedding_call = 0
        self.last_llm_call = 0
        self.embedding_delay = 3.0  # 3 seconds between calls
        self.llm_delay = 2.0  # 2 seconds between LLM calls

        # Fallback search data (when no embeddings available)
        self.document_chunks = []

    async def initialize(self):
        """Initialize the RAG service by loading and processing documents"""
        try:
            # Load documents
            documents_path = Path("data/documents")
            raw_documents = await self.document_loader.load_documents(documents_path)

            if not raw_documents:
                logger.warning("No documents found. Creating sample documents...")
                await self._create_sample_documents(documents_path)
                raw_documents = await self.document_loader.load_documents(
                    documents_path
                )

            # Split documents into chunks
            all_chunks = []
            for doc in raw_documents:
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update(
                        {
                            "chunk_id": f"{doc.metadata['source']}_{i}",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        }
                    )
                all_chunks.extend(chunks)

            logger.info(
                f"Created {len(all_chunks)} chunks from {len(raw_documents)} documents"
            )

            # Store chunks for fallback text search
            self.document_chunks = all_chunks

            # Try to load existing embeddings first
            if await self.vector_store.load_from_disk():
                logger.info(
                    "✅ Loaded existing embeddings from disk - no API calls needed!"
                )
            elif self.openai_available:
                # Only try to generate embeddings if OpenAI is available and we don't have existing ones
                logger.info(
                    "⚠️ No existing embeddings found. Attempting to generate (this may fail with free tier limits)..."
                )
                try:
                    await self.vector_store.add_documents(
                        all_chunks, self._get_embeddings
                    )
                    logger.info("✅ Successfully generated and saved embeddings!")
                except Exception as e:
                    if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                        logger.warning(
                            "⚠️ OpenAI quota exceeded - falling back to text search mode"
                        )
                        self.openai_available = False
                    else:
                        logger.error(f"Embedding generation failed: {e}")
                        self.openai_available = False
            else:
                logger.info(
                    "📝 Running in TEXT SEARCH MODE (no embeddings) - basic functionality available"
                )

            # Store document metadata
            for doc in raw_documents:
                chunks_for_doc = [
                    c
                    for c in all_chunks
                    if c.metadata["source"] == doc.metadata["source"]
                ]
                self.documents_metadata.append(
                    {
                        "filename": os.path.basename(doc.metadata["source"]),
                        "chunk_count": len(chunks_for_doc),
                        "file_size": doc.metadata.get("file_size", 0),
                        "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            self.is_initialized = True

            # Log the current mode
            if self.openai_available and len(self.vector_store.embeddings) > 0:
                logger.info("🚀 RAG service initialized in FULL MODE (embeddings + LLM)")
            elif self.openai_available:
                logger.info(
                    "🚀 RAG service initialized in LLM-ONLY MODE (text search + LLM)"
                )
            else:
                logger.info(
                    "🚀 RAG service initialized in BASIC MODE (text search only)"
                )

        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            # Don't raise the error - let it run in basic mode
            self.is_initialized = True
            logger.info("🚀 RAG service initialized in FALLBACK MODE")

    async def query(self, question: str, max_chunks: int = 3) -> QueryResponse:
        """Process a question using available methods"""
        start_time = time.time()

        try:
            # Get relevant chunks using best available method
            if self.openai_available and len(self.vector_store.embeddings) > 0:
                # Full embedding-based search
                similar_chunks = await self._embedding_search(question, max_chunks)
            elif len(self.document_chunks) > 0:
                # Fallback text-based search
                similar_chunks = await self._text_search(question, max_chunks)
            else:
                similar_chunks = []

            # Generate answer using best available method
            if self.openai_available and similar_chunks:
                try:
                    await self._rate_limit_llm()
                    answer = await self._generate_answer_with_llm(
                        question, similar_chunks
                    )
                except Exception as e:
                    if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                        logger.warning("LLM quota exceeded, using template response")
                        answer = self._generate_template_answer(
                            question, similar_chunks
                        )
                        self.openai_available = False  # Disable for future requests
                    else:
                        answer = self._generate_template_answer(
                            question, similar_chunks
                        )
            else:
                # Template-based response
                answer = self._generate_template_answer(question, similar_chunks)

            # Prepare retrieved chunks for response
            retrieved_chunks = [
                RetrievedChunk(
                    content=(
                        chunk.page_content
                        if hasattr(chunk, "page_content")
                        else chunk["content"]
                    ),
                    source=(
                        chunk.metadata["source"]
                        if hasattr(chunk, "metadata")
                        else chunk["metadata"]["source"]
                    ),
                    score=chunk.get("score", 0.5) if isinstance(chunk, dict) else 0.5,
                    chunk_id=(
                        chunk.metadata["chunk_id"]
                        if hasattr(chunk, "metadata")
                        else chunk["metadata"]["chunk_id"]
                    ),
                )
                for chunk in similar_chunks[:max_chunks]
            ]

            processing_time = time.time() - start_time

            return QueryResponse(
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                total_chunks_found=len(similar_chunks),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")

            # Provide helpful fallback response
            fallback_answer = f"""I apologize, but I'm currently running in limited mode due to API constraints.

Your question: "{question}"

**Current Status:** 
- OpenAI API: {'✅ Available' if self.openai_available else '❌ Quota exceeded or unavailable'}
- Embeddings: {'✅ Loaded' if len(getattr(self.vector_store, 'embeddings', [])) > 0 else '❌ Not available'}
- Documents: {'✅ Loaded' if len(self.document_chunks) > 0 else '❌ Not loaded'}

**What you can do:**
1. Wait a few hours for your OpenAI quota to reset
2. Add credits to your OpenAI account
3. Try simpler questions that don't require complex reasoning

The system has loaded {len(self.document_chunks)} document chunks about AI Safety and can still provide basic information using text search."""

            return QueryResponse(
                answer=fallback_answer,
                retrieved_chunks=[],
                total_chunks_found=0,
                processing_time=time.time() - start_time,
            )

    async def _embedding_search(self, question: str, max_chunks: int) -> List[Dict]:
        """Search using embeddings (premium mode)"""
        try:
            await self._rate_limit_embedding()
            question_embedding = await self._get_embeddings([question])

            similar_chunks = await self.vector_store.similarity_search(
                question_embedding[0], k=max_chunks * 2
            )

            # Rerank if Cohere is available
            if self.cohere_available and similar_chunks:
                try:
                    reranked = await self._rerank_chunks(question, similar_chunks)
                    return reranked[:max_chunks]
                except:
                    pass

            return similar_chunks[:max_chunks]

        except Exception as e:
            logger.warning(f"Embedding search failed: {e}")
            return await self._text_search(question, max_chunks)

    async def _text_search(self, question: str, max_chunks: int) -> List[Document]:
        """Fallback text-based search (free mode)"""
        if not self.document_chunks:
            return []

        logger.info("Using text-based search (no embeddings available)")

        # Simple keyword matching
        question_words = set(question.lower().split())

        scored_chunks = []
        for chunk in self.document_chunks:
            content = chunk.page_content.lower()

            # Simple scoring based on keyword matches
            matches = sum(
                1 for word in question_words if len(word) > 3 and word in content
            )
            word_density = matches / len(question_words) if question_words else 0

            # Boost score for chunks with question words in titles/headers
            if any(word in content[:100] for word in question_words):
                word_density += 0.3

            scored_chunks.append((word_density, chunk))

        # Sort by score and return top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[: max_chunks * 2]]

    def _generate_template_answer(self, question: str, chunks: List) -> str:
        """Generate answer using templates when LLM is not available"""
        if not chunks:
            return f"""I understand you're asking about: "{question}"

Unfortunately, I don't have access to the AI services needed to provide a detailed answer right now. However, I have loaded information about AI Safety and Ethics.

To get better answers:
1. Ensure your OpenAI API key has available credits
2. Wait for quota limits to reset
3. Try asking more specific questions

Topics I have information about:
- AI Alignment and Value Learning
- AI Safety Principles and Robustness
- AI Ethics and Governance
- Constitutional AI approaches
- Risk Assessment frameworks"""

        # Extract relevant content from chunks
        relevant_content = []
        for chunk in chunks[:3]:
            content = (
                chunk.page_content
                if hasattr(chunk, "page_content")
                else chunk.get("content", "")
            )
            source = (
                chunk.metadata.get("source", "")
                if hasattr(chunk, "metadata")
                else chunk.get("metadata", {}).get("source", "")
            )

            # Get first few sentences
            sentences = content.split(". ")[:3]
            excerpt = ". ".join(sentences)
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."

            relevant_content.append(
                {
                    "excerpt": excerpt,
                    "source": os.path.basename(source) if source else "Unknown",
                }
            )

        # Build template response
        response_parts = [
            f'Based on the available documents about AI Safety, here\'s what I found regarding "{question}":',
            "",
        ]

        for i, content in enumerate(relevant_content, 1):
            response_parts.append(f"**From {content['source']}:**")
            response_parts.append(content["excerpt"])
            response_parts.append("")

        response_parts.extend(
            [
                "**Note:** This response was generated using basic text search due to API limitations. For more sophisticated analysis, please ensure your OpenAI API key has available credits.",
                "",
                "**Available Topics:** AI Alignment, Safety Principles, Ethics, Constitutional AI, Risk Assessment",
            ]
        )

        return "\n".join(response_parts)

    async def _generate_answer_with_llm(self, question: str, chunks: List) -> str:
        """Generate answer using LLM when available"""
        try:
            # Prepare context
            context_parts = []
            for chunk in chunks[:3]:  # Limit to save tokens
                content = (
                    chunk.page_content
                    if hasattr(chunk, "page_content")
                    else chunk.get("content", "")
                )
                source = (
                    chunk.metadata.get("source", "")
                    if hasattr(chunk, "metadata")
                    else chunk.get("metadata", {}).get("source", "")
                )

                chunk_text = f"Source: {os.path.basename(source)}\n{content[:400]}"  # Limit chunk size
                context_parts.append(chunk_text)

            context = "\n\n".join(context_parts)

            prompt = f"""Based on the AI Safety documents, answer this question briefly and clearly:

Question: {question}

Context:
{context}

Provide a focused answer in under 150 words, citing sources when relevant."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,  # Very limited for free tier
                timeout=30.0,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning(f"LLM quota exceeded: {e}")
                raise  # Let caller handle this
            else:
                logger.error(f"LLM generation failed: {e}")
                return self._generate_template_answer(question, chunks)

    async def get_document_info(self) -> List[DocumentResponse]:
        """Get information about ingested documents"""
        return [DocumentResponse(**doc_meta) for doc_meta in self.documents_metadata]

    async def _rate_limit_embedding(self):
        """Rate limiting for embedding API calls"""
        now = time.time()
        time_since_last = now - self.last_embedding_call
        if time_since_last < self.embedding_delay:
            wait_time = self.embedding_delay - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s for embedding API")
            await asyncio.sleep(wait_time)
        self.last_embedding_call = time.time()

    async def _rate_limit_llm(self):
        """Rate limiting for LLM API calls"""
        now = time.time()
        time_since_last = now - self.last_llm_call
        if time_since_last < self.llm_delay:
            wait_time = self.llm_delay - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s for LLM API")
            await asyncio.sleep(wait_time)
        self.last_llm_call = time.time()

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI with heavy rate limiting"""
        if not self.openai_available:
            raise Exception("OpenAI not available")

        try:
            # Process one text at a time to minimize quota usage
            embeddings = []
            for i, text in enumerate(texts):
                if i > 0:  # Add delay between individual embeddings
                    await asyncio.sleep(2)

                response = await self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[text],  # Single text at a time
                )
                embeddings.append(response.data[0].embedding)

            return embeddings

        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.error(f"OpenAI quota/rate limit exceeded: {e}")
                self.openai_available = False
            raise

    async def _rerank_chunks(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank chunks using Cohere if available"""
        if not self.cohere_available:
            return chunks

        try:
            documents = [chunk["content"] for chunk in chunks]

            response = await self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=question,
                documents=documents,
                top_k=min(len(documents), 3),
            )

            reranked_chunks = []
            for result in response.results:
                chunk = chunks[result.index].copy()
                chunk["score"] = result.relevance_score
                reranked_chunks.append(chunk)

            return reranked_chunks

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return chunks

    async def _create_sample_documents(self, documents_path: Path):
        """Create focused sample documents for free tier"""
        documents_path.mkdir(parents=True, exist_ok=True)

        # Very focused, smaller documents
        sample_docs = {
            "ai_alignment.md": """# AI Alignment Basics

AI alignment ensures AI systems pursue goals aligned with human values.

## Core Challenge
Specifying what we want AI to do is difficult. Misaligned objectives can cause unintended consequences.

## Key Approaches
- Value Learning: AI learns human values through feedback
- Constitutional AI: AI follows guiding principles
- Robustness: Safe performance in unexpected situations

## Current Focus
Researchers work on reward modeling, interpretability, and safe exploration.
""",
            "ai_safety.md": """# AI Safety Principles

## Core Principles

**Robustness**: Reliable performance and graceful failures
**Human Oversight**: Meaningful human control over AI systems  
**Transparency**: Interpretable decision-making
**Fairness**: Treating all individuals and groups fairly
**Privacy**: Protecting personal data and rights
**Accountability**: Clear responsibility for AI behavior

## Risk Management
- Extensive testing before deployment
- Gradual rollout with monitoring
- Continuous assessment of performance
- Incident response procedures
""",
"ai_ethics.md": """# AI Ethics Overview

## Ethical Frameworks
- **Consequentialist**: Judge by outcomes and consequences
- **Deontological**: Focus on inherent rightness of actions
- **Virtue Ethics**: Emphasize character and virtues

## Key Issues

**Bias**: AI can perpetuate social biases
**Privacy**: Surveillance and data collection concerns  
**Autonomy**: Impact on human decision-making
**Employment**: Job displacement from automation

## Governance
Effective AI governance requires collaboration between government, industry, academia, and civil society.
""",
        }

        for filename, content in sample_docs.items():
            file_path = documents_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        logger.info(
            f"Created {len(sample_docs)} sample documents optimized for free tier"
        )
