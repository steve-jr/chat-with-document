# vector_store_pinecone.py - Pinecone Vector Store Implementation
import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
import hashlib
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    """Manages embeddings and vector search using Pinecone"""
    
    def __init__(self, 
                 session_id: str,
                 index_name: str = "company-chatbot",
                 namespace: str = "default",
                 embedding_model: str = 'all-MiniLM-L6-v2',
                ):    
        
        # Initialize Pinecone
        self.__api_key = os.environ.get('PINECONE_API_KEY')
        if not self.__api_key:
            raise ValueError("Pinecone API key not provided")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.__api_key)
        
        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.namespace = namespace
        self.session_id = session_id
        
        # Initialize or get index
        self.index = self._initialize_index(index_name)
        
        # Store for document tracking
        self.documents = {}
        
    def _initialize_index(self, index_name: str):
        """Initialize Pinecone index"""
        index_name = index_name.lower()
        try:
            # Check if index exists
            if not self.pc.has_index(index_name):
                logger.info(f"Creating new Pinecone index: {index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1',
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
                    
            # Connect to index
            index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
            return index
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to Pinecone index"""
        try:
            logger.info(f"Adding {len(documents)} documents to Pinecone")
            
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            
            # Prepare vectors for Pinecone
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                print(f"Processing document metadata {doc.metadata}")
                # Generate unique ID
                doc_id = self._generate_doc_id(doc.page_content, i)
                
                # Prepare metadata
                metadata = {
                    'text': doc.page_content[:1000],  # Pinecone has metadata size limits
                    'source': doc.metadata.get('source', 'unknown'),
                    'doc_type': doc.metadata.get('doc_type', 'general'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'session_id': self.session_id,
                    'namespace': self.namespace,
                    'created_at': doc.metadata.get('created_at', datetime.now().isoformat()),
                }
                
                # Store full document locally
                self.documents[doc_id] = doc
                
                vectors.append({
                    'id': doc_id,
                    'values': embedding.tolist(),
                    'metadata': metadata
                })
            
            # Upload in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            # Wait for indexing to complete
            time.sleep(1)
            
            logger.info(f"Successfully added {len(documents)} documents to namespace '{self.namespace}' for session '{self.session_id}'")
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for relevant documents in Pinecone"""
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode([query])[0]
            filter = {}
            if self.session_id:
                filter["session_id"] = {"$eq": self.session_id}

            # Search in Pinecone by session ID and namespace
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                namespace=self.namespace,
                include_metadata=True,
                filter=filter
            )
            
            # Process results
            search_results = []
            for match in results['matches']:
                doc_id = match['id']
                score = match['score']
                
                # Try to get full document from local storage
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                else:
                    # Reconstruct from metadata if not in local storage
                    metadata = match.get('metadata', {})
                    doc = Document(
                        page_content=metadata.get('text', ''),
                        metadata={
                            'source': metadata.get('source', 'unknown'),
                            'doc_type': metadata.get('doc_type', 'general')
                        }
                    )
                
                search_results.append((doc, score))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching in Pinecone: {str(e)}")
            return []
    
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.get('namespaces', {}).get(self.namespace, {})
            
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'namespace_vectors': namespace_stats.get('vector_count', 0),
                'dimension': stats.get('dimension', self.dimension),
                'index_fullness': stats.get('index_fullness', 0),
                'namespaces': list(stats.get('namespaces', {}).keys())
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def _generate_doc_id(self, content: str, index: int) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{self.namespace}_{content_hash}_{index}"