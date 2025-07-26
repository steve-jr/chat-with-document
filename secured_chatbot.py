from typing import Dict, List, Tuple, Optional
from chatbot import RAGChatbot
from security_filter import SecurityFilter
from datetime import datetime
import hashlib


class SecureRAGChatbot(RAGChatbot):
    """Extended chatbot with security filtering"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.security_filter = SecurityFilter()
    
    def get_response(self, query: str, k: int = 5) -> Dict[str, any]:
        """Get response with security filtering"""
        
        # Check query safety first
        is_safe, reason = self.security_filter.is_query_safe(query)
        if not is_safe:
            self.security_filter.log_security_event("UNSAFE_QUERY", {
                "reason": reason,
                "risk_level": self.security_filter.get_risk_level(query)
            })
            
            return {
                "response": "I notice your query contains sensitive information. For security reasons, please don't share personal details like account numbers, SSN, or passwords. How can I help you with general company information?",
                "sources": [],
                "query_id": hashlib.md5(f"{query}{datetime.now()}".encode()).hexdigest()[:8],
                "confidence": 0.0,
                "security_flag": True
            }
        
        # Check if human review needed
        if self.security_filter.requires_human_review(query):
            self.security_filter.log_security_event("HUMAN_REVIEW_REQUIRED", {
                "query_preview": query[:50],
                "risk_level": self.security_filter.get_risk_level(query)
            })
        
        # Get normal response
        response_data = super().get_response(query, k)
        
        # Sanitize response before returning
        response_data["response"] = self.security_filter.sanitize_response(response_data["response"])
        
        return response_data