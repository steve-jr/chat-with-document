import re
from typing import List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

class SecurityFilter:
    """Handles security filtering for queries and responses"""
    
    def __init__(self):
        # PII patterns - common patterns that might appear in banking queries
        self.pii_patterns = {
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',  # Social Security Number
            "credit_card": r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',  # Credit card
            "account_number": r'\b\d{9,12}\b',  # Bank account numbers
            "routing_number": r'\b\d{9}\b',  # Routing numbers
            "phone": r'\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b',  # Phone numbers
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            "drivers_license": r'\b[A-Z]{1,2}\d{5,8}\b',  # Driver's license (varies by state)
            "passport": r'\b[A-Z][0-9]{8}\b',  # US Passport format
        }
        
        # Sensitive topics that require special handling
        self.sensitive_topics = [
            "fraud", "hack", "breach", "lawsuit", "complaint",
            "bankruptcy", "foreclosure", "collection", "debt",
            "suicide", "death", "divorce", "emergency",
            "investigation", "audit", "compliance violation"
        ]
        
        # Blocked terms - should never be requested or stored
        self.blocked_terms = [
            "password", "pin", "cvv", "security code",
            "secret question", "mother's maiden name",
            "full ssn", "complete social"
        ]
        
        # High-risk queries that need logging
        self.high_risk_patterns = [
            r'transfer.*all.*money',
            r'close.*all.*accounts',
            r'withdraw.*everything',
            r'give.*access.*account',
            r'share.*login.*credentials'
        ]
    
    def check_pii(self, text: str) -> List[Tuple[str, str]]:
        """Check text for PII patterns"""
        found_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_pii.extend([(pii_type, match) for match in matches])
                logger.warning(f"PII detected - Type: {pii_type}, Count: {len(matches)}")
        
        return found_pii
    
    def is_query_safe(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query is safe to process"""
        
        # Check for PII
        pii_found = self.check_pii(query)
        if pii_found:
            pii_types = list(set([pii[0] for pii in pii_found]))
            return False, f"Query contains sensitive information: {', '.join(pii_types)}"
        
        # Check for blocked terms
        query_lower = query.lower()
        for term in self.blocked_terms:
            if term in query_lower:
                logger.warning(f"Blocked term detected: {term}")
                return False, f"Query contains restricted information"
        
        # Check for high-risk patterns
        for pattern in self.high_risk_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"High-risk query pattern detected: {pattern}")
                # Don't block, but flag for review
        
        return True, None
    
    def sanitize_response(self, response: str) -> str:
        """Sanitize response to ensure no PII is exposed"""
        sanitized = response
        
        # Replace any PII patterns found in response
        for pii_type, pattern in self.pii_patterns.items():
            # Count matches before replacing
            matches = re.findall(pattern, sanitized)
            if matches:
                logger.warning(f"Sanitizing {len(matches)} {pii_type} patterns from response")
            
            # Replace with safe placeholder
            sanitized = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def requires_human_review(self, query: str) -> bool:
        """Check if query involves sensitive topics requiring human review"""
        query_lower = query.lower()
        
        for topic in self.sensitive_topics:
            if topic in query_lower:
                logger.info(f"Sensitive topic detected, flagging for review: {topic}")
                return True
        
        # Check for high-risk patterns
        for pattern in self.high_risk_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"High-risk pattern detected, flagging for review")
                return True
        
        return False
    
    def get_risk_level(self, query: str) -> str:
        """Assess risk level of query"""
        query_lower = query.lower()
        
        # Critical risk - contains PII or blocked terms
        if self.check_pii(query) or any(term in query_lower for term in self.blocked_terms):
            return "CRITICAL"
        
        # High risk - contains sensitive topics or high-risk patterns
        if (any(topic in query_lower for topic in self.sensitive_topics) or 
            any(re.search(pattern, query_lower) for pattern in self.high_risk_patterns)):
            return "HIGH"
        
        # Medium risk - contains financial amounts or account references
        if re.search(r'\$[\d,]+', query) or 'account' in query_lower:
            return "MEDIUM"
        
        return "LOW"
    
    def log_security_event(self, event_type: str, details: dict):
        """Log security-related events for audit trail"""
        logger.warning(f"SECURITY_EVENT: {event_type} - {json.dumps(details)}")