import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from openai import OpenAI

logger = logging.getLogger(__name__)

class RAGChatbot:
    """Main RAG chatbot for company queries"""
    
    def __init__(self, 
                 vector_store,
                 model: str = "gpt-3.5-turbo"):
        
        self.vector_store = vector_store
        self.model = model
        
        # Initialize OpenAI client
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OpenAI API key not provided")
        
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

        self.client = OpenAI()
        
        # Conversation history for this session
        self.conversation_history = []
        
    def get_response(self, query: str, k: int = 5) -> Dict[str, any]:
        """Generate response using RAG pipeline"""
        
        # Generate query ID for tracking
        query_id = hashlib.md5(f"{query}{datetime.now()}".encode()).hexdigest()[:8]
        logger.info(f"Query [{query_id}]: {query[:50]}...")
        
        try:
            # Retrieve relevant chunks
            search_results = self.vector_store.search(query, k=k)
            
            if not search_results:
                return {
                    "response": "I couldn't find relevant information in the uploaded documents. Please make sure you've uploaded the appropriate company documents.",
                    "sources": [],
                    "query_id": query_id,
                    "confidence": 0.0
                }
            
            # Extract chunks and sources
            context_chunks = [result[0] for result in search_results]
            sources = list(set([chunk.metadata.get('source', 'Unknown') for chunk in context_chunks]))
            confidence = search_results[0][1] if search_results else 0.0
            
            # Build prompt
            prompt = self._build_prompt(query, context_chunks)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful company assistant. Only answer based on the provided context. If the information is not in the context, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": answer
            })
            
            # Keep only last 10 interactions
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Response [{query_id}]: Generated successfully")
            
            return {
                "response": answer,
                "sources": sources,
                "query_id": query_id,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error generating response [{query_id}]: {str(e)}")
            return {
                "response": "I'm experiencing technical difficulties. Please try again later.",
                "sources": [],
                "query_id": query_id,
                "confidence": 0.0
            }
    
    def _build_prompt(self, query: str, context_chunks: List) -> str:
        """Construct prompt with retrieved context"""
        
        # Include recent conversation history if available
        history_context = ""
        if len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-3:]  # Last 3 interactions
            history_parts = []
            for h in recent_history:
                history_parts.append(f"Customer: {h['query']}")
                history_parts.append(f"Assistant: {h['response']}")
            history_context = "Recent conversation:\n" + "\n".join(history_parts) + "\n\n"
        
        # Combine context chunks
        context = "\n\n".join([
            f"[Source: {chunk.metadata.get('doc_type', 'unknown')}]\n{chunk.page_content}"
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a helpful company assistant. Your role is to provide accurate, compliant information based solely on the company's official documentation provided below.

IMPORTANT INSTRUCTIONS:
1. Only answer based on the provided context
2. If the answer is not in the context, say "I don't have that information in the uploaded documents"
3. Be friendly but professional
4. Never make up information
5. For sensitive topics, remind customers to visit a branch or call customer service
6. Use simple, clear language
7. Consider the conversation history when relevant

{history_context}

CONTEXT FROM COMPANY DOCUMENTATION:
{context}

CURRENT CUSTOMER QUESTION: {query}

RESPONSE:"""
        
        return prompt
