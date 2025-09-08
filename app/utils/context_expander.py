# app/utils/context_expander.py
from typing import List, Dict

try:
    from .logger import logger
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False
    class logger:
        @staticmethod
        def log_app_event(event, details=None, level="info"): pass

class ContextExpander:
    """Expand search results with surrounding context chunks"""
    
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
    
    def get_surrounding_chunks(self, chunk_metadata: Dict, context_window: int = 1) -> List[Dict]:
        """Get chunks before and after the current chunk from the same sermon"""
        try:
            filename = chunk_metadata.get('filename')
            current_index = chunk_metadata.get('chunk_index', 0)
            
            if not filename:
                return []
            
            all_results = self.ai_engine.collection.get(where={"filename": filename})
            
            if not all_results or not all_results['metadatas']:
                return []
            
            chunks_with_index = []
            for i, (doc_id, doc, metadata) in enumerate(zip(
                all_results['ids'], 
                all_results['documents'], 
                all_results['metadatas']
            )):
                chunks_with_index.append({
                    'id': doc_id,
                    'content': doc,
                    'metadata': metadata,
                    'index': metadata.get('chunk_index', 0)
                })
            
            chunks_with_index.sort(key=lambda x: x['index'])
            
            surrounding_chunks = []
            for chunk in chunks_with_index:
                chunk_idx = chunk['index']
                if (current_index - context_window <= chunk_idx <= current_index + context_window 
                    and chunk_idx != current_index):
                    surrounding_chunks.append(chunk)
            
            return surrounding_chunks
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("context_expansion_failed", {"error": str(e)}, level="warning")
            return []
    
    def expand_chunk_context(self, chunk: Dict, context_window: int = 1) -> Dict:
        """Expand a single chunk with surrounding context"""
        original_content = chunk['content']
        metadata = chunk['metadata']
        
        surrounding_chunks = self.get_surrounding_chunks(metadata, context_window)
        
        if not surrounding_chunks:
            return chunk
        
        surrounding_chunks.sort(key=lambda x: x['metadata']['chunk_index'])
        
        expanded_parts = []
        current_index = metadata.get('chunk_index', 0)
        
        # Add preceding chunks
        for surrounding_chunk in surrounding_chunks:
            chunk_idx = surrounding_chunk['metadata']['chunk_index']
            if chunk_idx < current_index:
                expanded_parts.append(f"[Previous context]: {surrounding_chunk['content'][:200]}...")
        
        # Add main chunk
        expanded_parts.append(f"[Main content]: {original_content}")
        
        # Add following chunks
        for surrounding_chunk in surrounding_chunks:
            chunk_idx = surrounding_chunk['metadata']['chunk_index']
            if chunk_idx > current_index:
                expanded_parts.append(f"[Following context]: {surrounding_chunk['content'][:200]}...")
        
        expanded_chunk = chunk.copy()
        expanded_chunk['expanded_content'] = "\n\n".join(expanded_parts)
        expanded_chunk['original_content'] = original_content
        expanded_chunk['context_added'] = len(surrounding_chunks)
        
        return expanded_chunk
    
    def expand_search_results(self, results: List[Dict], context_window: int = 1, 
                            max_expansions: int = 3) -> List[Dict]:
        """Expand multiple search results with context"""
        expanded_results = []
        expansions_done = 0
        
        for result in results:
            if expansions_done < max_expansions:
                relevance_threshold = 0.7
                if (result.get('relevance_score', 0) > relevance_threshold or 
                    result.get('combined_score', 0) > relevance_threshold):
                    
                    expanded_result = self.expand_chunk_context(result, context_window)
                    expanded_results.append(expanded_result)
                    expansions_done += 1
                else:
                    expanded_results.append(result)
            else:
                expanded_results.append(result)
        
        return expanded_results