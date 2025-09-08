# app/utils/relevance_reranker.py
import re
from typing import List, Dict, Any

try:
    from .logger import logger
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False
    class logger:
        @staticmethod
        def log_app_event(event, details=None, level="info"): pass

class RelevanceReranker:
    """Re-rank search results based on multiple relevance factors"""
    
    def __init__(self):
        self.theological_terms = {
            'salvation', 'grace', 'faith', 'prayer', 'worship', 'scripture', 'bible',
            'jesus', 'christ', 'god', 'holy', 'spirit', 'lord', 'gospel', 'church',
            'love', 'forgiveness', 'sin', 'redemption', 'covenant', 'kingdom',
            'discipleship', 'ministry', 'pastor', 'preaching', 'teaching'
        }
        
        self.question_patterns = {
            'what': ['what is', 'what does', 'what means'],
            'how': ['how to', 'how can', 'how do'],
            'why': ['why is', 'why does', 'why should'],
            'when': ['when did', 'when should', 'when will'],
            'who': ['who is', 'who was', 'who are']
        }
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the user's query to understand intent"""
        query_lower = query.lower()
        
        intent = {
            'question_type': None,
            'theological_focus': False,
            'biblical_reference': False,
            'practical_application': False
        }
        
        # Detect question type
        for qtype, patterns in self.question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                intent['question_type'] = qtype
                break
        
        # Check for theological terms
        query_words = set(query_lower.split())
        theological_matches = query_words.intersection(self.theological_terms)
        intent['theological_focus'] = len(theological_matches) > 0
        intent['theological_terms'] = list(theological_matches)
        
        # Check for biblical references
        biblical_patterns = [
            r'\b\d+\s*\w+\s*\d+',
            r'\b(?:genesis|exodus|matthew|john|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|hebrews|james|peter|revelation)\b'
        ]
        
        for pattern in biblical_patterns:
            if re.search(pattern, query_lower):
                intent['biblical_reference'] = True
                break
        
        # Check for practical application keywords
        practical_keywords = ['how to', 'apply', 'practice', 'daily', 'life', 'example']
        intent['practical_application'] = any(keyword in query_lower for keyword in practical_keywords)
        
        return intent
    
    def score_chunk_relevance(self, chunk: Dict, query: str, intent: Dict[str, Any]) -> float:
        """Score a chunk's relevance based on multiple factors"""
        content = chunk['content'].lower()
        metadata = chunk['metadata']
        
        relevance_score = 0.0
        
        # Base semantic score
        relevance_score += chunk.get('combined_score', chunk.get('semantic_score', 0.0)) * 0.4
        
        # Query term density bonus
        query_terms = query.lower().split()
        term_matches = sum(1 for term in query_terms if term in content)
        term_density = term_matches / len(query_terms) if query_terms else 0
        relevance_score += term_density * 0.2
        
        # Theological relevance bonus
        if intent['theological_focus']:
            theological_matches = sum(1 for term in intent.get('theological_terms', []) 
                                    if term in content)
            relevance_score += min(theological_matches * 0.1, 0.3)
        
        # Question type matching
        if intent['question_type']:
            question_indicators = {
                'what': ['definition', 'meaning', 'is', 'means'],
                'how': ['steps', 'process', 'method', 'way'],
                'why': ['because', 'reason', 'purpose', 'therefore'],
                'when': ['time', 'during', 'after', 'before'],
                'who': ['person', 'people', 'character']
            }
            
            indicators = question_indicators.get(intent['question_type'], [])
            indicator_matches = sum(1 for indicator in indicators if indicator in content)
            relevance_score += min(indicator_matches * 0.05, 0.15)
        
        # Biblical reference bonus
        if intent['biblical_reference']:
            if re.search(r'\b\d+:\d+\b', content):
                relevance_score += 0.15
        
        # Chunk position bonus (earlier chunks often have main points)
        chunk_index = metadata.get('chunk_index', 0)
        if chunk_index <= 2:
            relevance_score += 0.05
        
        return min(relevance_score, 1.0)
    
    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Re-rank results based on relevance scoring"""
        try:
            intent = self.analyze_query_intent(query)
            
            for result in results:
                relevance_score = self.score_chunk_relevance(result, query, intent)
                result['relevance_score'] = relevance_score
                result['query_intent'] = intent
            
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("reranking_failed", {"error": str(e)}, level="warning")
            return results