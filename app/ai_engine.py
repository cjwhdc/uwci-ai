import streamlit as st
import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# Fix for SQLite version issue on Streamlit Cloud
try:
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
import requests
from openai import OpenAI

from .sermon_processor import SermonProcessor

# Import enhanced components
try:
    from .utils.relevance_reranker import RelevanceReranker
    from .utils.context_expander import ContextExpander
    from .utils.metadata_filter import MetadataFilter
    SEARCH_ENHANCEMENTS = True
except ImportError:
    SEARCH_ENHANCEMENTS = False

# Import enhanced utilities if available
try:
    from .utils.error_handler import handle_errors, ErrorHandler, AIModelError, DatabaseError
    from .utils.logger import logger, log_ai_operation_decorator, log_db_operation_decorator
    ENHANCED_FEATURES = True
except ImportError:
    # Fallback for basic functionality
    ENHANCED_FEATURES = False
    def handle_errors(context="", user_message=""):
        def decorator(func):
            return func
        return decorator
    
    class logger:
        @staticmethod
        def log_user_activity(activity, details=None): pass
        @staticmethod
        def log_db_operation(op, success=True, details=None): pass
        @staticmethod
        def log_ai_operation(op, model, success=True, **kwargs): pass
        @staticmethod
        def log_app_event(event, details=None, level="info"): pass
    
    def log_ai_operation_decorator(model):
        def decorator(func):
            return func
        return decorator
    
    def log_db_operation_decorator(op):
        def decorator(func):
            return func
        return decorator

class AIEngine:
    """Enhanced AI processing engine with improved search capabilities"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        
        # Enhanced components (only if available)
        if SEARCH_ENHANCEMENTS:
            self.reranker = RelevanceReranker()
            self.context_expander = ContextExpander(self)
            self.metadata_filter = MetadataFilter()
        
        # Query cache for performance
        self._query_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        # Keyword index for hybrid search
        self._document_texts = {}
        self._keyword_index = {}
        
        if ENHANCED_FEATURES:
            self.error_handler = ErrorHandler()
        
        # Try to sync database from cloud first
        self.sync_database_from_cloud()
        
        self.setup_vector_db()
        
        # Build keyword index for hybrid search
        if SEARCH_ENHANCEMENTS:
            self._build_keyword_index()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'a', 'an', 'as', 'if', 'then', 'than', 'so',
            'very', 'just', 'now', 'only', 'also', 'even', 'well', 'back', 'still'
        }
        
        keywords = [word for word in words 
                   if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _build_keyword_index(self):
        """Build keyword index for fast text search"""
        try:
            if not self.collection:
                return
            
            results = self.collection.get()
            if results and results['documents']:
                for i, (doc_id, document) in enumerate(zip(results['ids'], results['documents'])):
                    self._document_texts[doc_id] = document
                    
                    words = self._extract_keywords(document)
                    for word in words:
                        if word not in self._keyword_index:
                            self._keyword_index[word] = set()
                        self._keyword_index[word].add(doc_id)
                        
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("keyword_index_build_failed", {"error": str(e)}, level="warning")
    
    def _keyword_search(self, query: str, max_results: int = 50) -> Dict[str, float]:
        """Perform keyword-based search and return scores"""
        query_keywords = self._extract_keywords(query)
        doc_scores = Counter()
        
        for keyword in query_keywords:
            if keyword in self._keyword_index:
                for doc_id in self._keyword_index[keyword]:
                    doc_scores[doc_id] += 1
        
        if doc_scores:
            max_score = max(doc_scores.values())
            # Normalize scores - keep as Counter until we get top results
            normalized_scores = {doc_id: score / max_score 
                            for doc_id, score in doc_scores.items()}
            
            # Get top results using Counter's most_common, then convert to dict
            top_items = doc_scores.most_common(max_results)
            top_docs = {doc_id: normalized_scores[doc_id] for doc_id, _ in top_items}
            
            return top_docs
        
        return {}
    
    def _combine_scores(self, semantic_results: List[Dict], keyword_scores: Dict[str, float], 
                       semantic_weight: float = 0.7) -> List[Dict]:
        """Combine semantic and keyword search scores"""
        keyword_weight = 1.0 - semantic_weight
        
        combined_results = []
        
        for result in semantic_results:
            doc_id = result['metadata'].get('chunk_id', '')
            
            semantic_score = 1.0 - result.get('distance', 0.5)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            
            combined_score = (semantic_weight * semantic_score + 
                            keyword_weight * keyword_score)
            
            result['combined_score'] = combined_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            
            combined_results.append(result)
        
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results
    
    @handle_errors(context="cloud_sync", user_message="Cloud sync temporarily unavailable")
    def sync_database_from_cloud(self):
        """Sync database from Dropbox if configured"""
        try:
            from app.config.config import USE_DROPBOX_SYNC, DropboxDatabaseSync
            if USE_DROPBOX_SYNC:
                sync = DropboxDatabaseSync()
                if sync.sync_from_dropbox():
                    if ENHANCED_FEATURES:
                        logger.log_app_event("dropbox_sync_success", {"direction": "download"})
                    return True
        except (ImportError, AttributeError):
            pass
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("dropbox_sync_failed", {"direction": "download", "error": str(e)}, level="warning")
            st.warning(f"Dropbox sync failed, using local database: {e}")
        return False
    
    @handle_errors(context="cloud_backup", user_message="Cloud backup temporarily unavailable")
    def sync_database_to_cloud(self):
        """Sync database to Dropbox if configured"""
        try:
            from app.config.config import USE_DROPBOX_SYNC, DropboxDatabaseSync
            if USE_DROPBOX_SYNC:
                sync = DropboxDatabaseSync()
                success = sync.sync_to_dropbox()
                if success:
                    if ENHANCED_FEATURES:
                        logger.log_app_event("dropbox_sync_success", {"direction": "upload"})
                else:
                    if ENHANCED_FEATURES:
                        logger.log_app_event("dropbox_sync_failed", {"direction": "upload"}, level="warning")
                return success
        except (ImportError, AttributeError):
            return False
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("dropbox_sync_failed", {"direction": "upload", "error": str(e)}, level="error")
            st.error(f"Dropbox sync failed: {e}")
            return False
    
    def get_cloud_backup_info(self):
        """Get information about cloud backup"""
        try:
            from app.config.config import USE_DROPBOX_SYNC, DropboxDatabaseSync
            if USE_DROPBOX_SYNC:
                sync = DropboxDatabaseSync()
                return sync.get_backup_info()
        except (ImportError, AttributeError):
            pass
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("backup_info_failed", {"error": str(e)}, level="error")
            st.error(f"Error getting backup info: {e}")
        return None
    
    @handle_errors(context="vector_db_setup", user_message="Database initialization failed")
    def setup_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.client = chromadb.PersistentClient(path="./sermon_db")
            
            self.collection = self.client.get_or_create_collection(
                name="sermons",
                metadata={"description": "Sermon transcripts and chunks"}
            )
            
            if ENHANCED_FEATURES:
                logger.log_db_operation("database_init", success=True, details={
                    "client_type": "persistent",
                    "collection_name": "sermons"
                })
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("database_init", success=False, details={"error": str(e)})
                raise DatabaseError(f"Error setting up vector database: {e}")
            else:
                st.error(f"Error setting up vector database: {e}")
    
    @log_db_operation_decorator("get_processed_files")
    def get_processed_files(self) -> set:
        """Get list of already processed sermon files"""
        try:
            results = self.collection.get()
            if results and results['metadatas']:
                processed = {meta.get('filename', '') for meta in results['metadatas']}
                return processed
            return set()
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("get_processed_files", success=False, details={"error": str(e)})
            st.error(f"Error getting processed files: {e}")
            return set()
    
    @handle_errors(context="sermon_processing", user_message="Sermon processing temporarily unavailable")
    def process_new_sermons(self):
        """Process new sermons from data/sermons folder"""
        sermons_dir = Path("data/sermons")
        
        if not sermons_dir.exists():
            st.warning("Sermons directory 'data/sermons' not found. Please create it and add your .md files.")
            return
        
        sermon_files = list(sermons_dir.glob("*.md"))
        
        if not sermon_files:
            st.info("No sermon files found in data/sermons folder. Add .md files to process them.")
            return
        
        processed_files = self.get_processed_files()
        new_files = [f for f in sermon_files if f.name not in processed_files]
        
        if new_files:
            st.info(f"Found {len(new_files)} new sermon(s) to process...")
            
            processor = SermonProcessor()
            progress_bar = st.progress(0)
            
            processed_count = 0
            failed_count = 0
            
            for i, sermon_file in enumerate(new_files):
                try:
                    st.write(f"Processing: {sermon_file.name}")
                    
                    with open(sermon_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    metadata = processor.extract_metadata(content, sermon_file.name)
                    chunks = processor.chunk_sermon(content)
                    
                    success = self.add_sermon(content, metadata, chunks)
                    
                    if success:
                        st.success(f"Successfully processed: {sermon_file.name}")
                        processed_count += 1
                        if ENHANCED_FEATURES:
                            logger.log_app_event("sermon_processed", {
                                "filename": sermon_file.name,
                                "chunks": len(chunks),
                                "word_count": metadata.get('word_count', 0)
                            })
                    else:
                        st.error(f"Failed to process: {sermon_file.name}")
                        failed_count += 1
                    
                    time.sleep(1)
                    progress_bar.progress((i + 1) / len(new_files))
                    
                except Exception as e:
                    st.error(f"Error processing {sermon_file.name}: {e}")
                    failed_count += 1
                    if ENHANCED_FEATURES:
                        logger.log_app_event("sermon_processing_failed", {
                            "filename": sermon_file.name,
                            "error": str(e)
                        }, level="error")
            
            # Rebuild keyword index after processing
            if SEARCH_ENHANCEMENTS:
                self._build_keyword_index()
            
            if processed_count > 0:
                st.success(f"Processing complete! Successfully processed {processed_count} sermon(s).")
            if failed_count > 0:
                st.warning(f"{failed_count} sermon(s) failed to process. Check logs for details.")
        else:
            st.info(f"All {len(sermon_files)} sermon(s) already processed. Database is up to date.")
    
    @handle_errors(context="text_embedding", user_message="Text processing temporarily unavailable")
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using sentence transformers"""
        if not hasattr(self, 'embedder') or self.embedder is None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                if ENHANCED_FEATURES:
                    logger.log_app_event("embedding_model_loaded", {"model": "all-MiniLM-L6-v2"})
            except Exception as e:
                if ENHANCED_FEATURES:
                    logger.log_app_event("embedding_model_failed", {"error": str(e)}, level="error")
                st.error(f"Error loading embedding model: {e}")
                return []
        
        try:
            embedding = self.embedder.encode(text).tolist()
            return embedding
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_app_event("embedding_failed", {"error": str(e)}, level="error")
            st.error(f"Error generating embedding: {e}")
            return []
    
    @log_db_operation_decorator("add_sermon")
    def add_sermon(self, content: str, metadata: Dict[str, Any], chunks: List[str]):
        """Add sermon to vector database"""
        try:
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{metadata['filename']}_{i}"
                chunk_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'chunk_id': doc_id
                }
                
                embedding = self.embed_text(chunk)
                if embedding:
                    documents.append(chunk)
                    metadatas.append(chunk_metadata)
                    ids.append(doc_id)
                    embeddings.append(embedding)
            
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                
                if ENHANCED_FEATURES:
                    logger.log_db_operation("add_sermon", success=True, details={
                        "filename": metadata['filename'],
                        "chunks_added": len(documents)
                    })
                return True
            return False
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("add_sermon", success=False, details={
                    "filename": metadata.get('filename', 'unknown'),
                    "error": str(e)
                })
            st.error(f"Error adding sermon to database: {e}")
            return False
    
    def _get_cache_key(self, query: str, pastor_filter: str = None) -> str:
        """Generate cache key for query"""
        cache_data = f"{query}|{pastor_filter or 'all'}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    @log_db_operation_decorator("search")
    def search_sermons(self, query: str, pastor_filter: str = None, n_results: int = None) -> List[Dict]:
        """Basic search for compatibility"""
        try:
            if ENHANCED_FEATURES:
                logger.log_user_activity("sermon_search", {
                    'query_length': len(query),
                    'pastor_filter': pastor_filter,
                    'max_results': n_results
                })
            
            # Check cache first
            cache_key = self._get_cache_key(query, pastor_filter)
            current_time = time.time()
            
            if cache_key in self._query_cache:
                cached_result, timestamp = self._query_cache[cache_key]
                if current_time - timestamp < self._cache_timeout:
                    if ENHANCED_FEATURES:
                        logger.log_db_operation("search_cached", success=True, details={
                            'cache_hit': True,
                            'query_hash': cache_key[:8]
                        })
                    return cached_result
            
            if n_results is None:
                n_results = st.session_state.get('max_search_results', 10)
            
            query_embedding = self.embed_text(query)
            if not query_embedding:
                return []
            
            where_filter = {}
            if pastor_filter and pastor_filter != "All Pastors":
                where_filter["pastor"] = pastor_filter
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None
            )
            
            search_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    }
                    search_results.append(result)
            
            # Cache results
            self._query_cache[cache_key] = (search_results, current_time)
            
            # Clean old cache entries
            if len(self._query_cache) > 100:
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
            
            if ENHANCED_FEATURES:
                logger.log_db_operation("search", success=True, details={
                    'results_count': len(search_results),
                    'query_hash': cache_key[:8],
                    'cached': False
                })
            
            return search_results
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("search", success=False, details={
                    'error': str(e),
                    'query_hash': self._get_cache_key(query, pastor_filter)[:8]
                })
            st.error(f"Error searching sermons: {e}")
            return []
    
    def enhanced_search_sermons(self, query: str, pastor_filter: str = None, 
                               n_results: int = None, use_hybrid: bool = True) -> List[Dict]:
        """Enhanced search with hybrid semantic + keyword matching (if available)"""
        if not SEARCH_ENHANCEMENTS:
            return self.search_sermons(query, pastor_filter, n_results)
        
        try:
            if ENHANCED_FEATURES:
                logger.log_user_activity("enhanced_search", {
                    'query_length': len(query),
                    'pastor_filter': pastor_filter,
                    'use_hybrid': use_hybrid
                })
            
            if n_results is None:
                n_results = st.session_state.get('max_search_results', 10)
            
            search_limit = n_results * 3 if use_hybrid else n_results
            semantic_results = self.search_sermons(query, pastor_filter, search_limit)
            
            if not use_hybrid or not semantic_results:
                return semantic_results[:n_results]
            
            keyword_scores = self._keyword_search(query, search_limit)
            combined_results = self._combine_scores(semantic_results, keyword_scores)
            
            final_results = combined_results[:n_results]
            
            if ENHANCED_FEATURES:
                logger.log_db_operation("enhanced_search", success=True, details={
                    'semantic_results': len(semantic_results),
                    'keyword_matches': len(keyword_scores),
                    'final_results': len(final_results)
                })
            
            return final_results
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("enhanced_search", success=False, details={'error': str(e)})
            return self.search_sermons(query, pastor_filter, n_results)
    
    def ultimate_search_sermons(self, query: str, pastor_filter: str = None, 
                               n_results: int = None, search_options: Dict = None) -> List[Dict]:
        """Ultimate search with all enhancements (if available)"""
        if not SEARCH_ENHANCEMENTS:
            return self.search_sermons(query, pastor_filter, n_results)
        
        try:
            default_options = {
                'use_hybrid': True,
                'use_reranking': True,
                'expand_context': True,
                'context_window': 1,
                'max_context_expansions': 3,
                'use_metadata_weighting': True,
                'user_preferences': None
            }
            
            if search_options:
                default_options.update(search_options)
            options = default_options
            
            if n_results is None:
                n_results = st.session_state.get('max_search_results', 10)
            
            if ENHANCED_FEATURES:
                logger.log_user_activity("ultimate_search", {
                    'query_length': len(query),
                    'pastor_filter': pastor_filter,
                    'options': options
                })
            
            # Step 1: Hybrid Search
            if options['use_hybrid']:
                results = self.enhanced_search_sermons(
                    query, pastor_filter, n_results * 2, use_hybrid=True
                )
            else:
                results = self.search_sermons(query, pastor_filter, n_results * 2)
            
            if not results:
                return []
            
            # Step 2: Relevance Re-ranking
            if options['use_reranking']:
                results = self.reranker.rerank_results(results, query)
            
            # Step 3: Metadata-based filtering and weighting
            if options['use_metadata_weighting']:
                filters = self.metadata_filter.create_advanced_filters(
                    query, options.get('user_preferences')
                )
                results = self.metadata_filter.apply_metadata_weighting(results, filters)
            
            # Step 4: Context expansion for top results
            if options['expand_context']:
                results = self.context_expander.expand_search_results(
                    results, options['context_window']
                )
            
            # Step 5: Final result selection and formatting
            final_results = results[:n_results]
            
            for i, result in enumerate(final_results):
                result['search_rank'] = i + 1
                result['search_method'] = 'enhanced_ultimate'
                
                confidence_factors = []
                if 'combined_score' in result:
                    confidence_factors.append(result['combined_score'])
                if 'relevance_score' in result:
                    confidence_factors.append(result['relevance_score'])
                if 'weighted_score' in result:
                    confidence_factors.append(result['weighted_score'])
                
                if confidence_factors:
                    result['confidence'] = sum(confidence_factors) / len(confidence_factors)
                else:
                    result['confidence'] = 0.5
            
            if ENHANCED_FEATURES:
                logger.log_db_operation("ultimate_search", success=True, details={
                    'query_hash': hashlib.md5(query.encode()).hexdigest()[:8],
                    'results_returned': len(final_results),
                    'avg_confidence': sum(r.get('confidence', 0) for r in final_results) / len(final_results) if final_results else 0,
                    'top_confidence': final_results[0].get('confidence', 0) if final_results else 0,
                    'search_options': options
                })
            
            return final_results
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("ultimate_search", success=False, details={
                    'error': str(e),
                    'fallback_used': True
                })
            return self.search_sermons(query, pastor_filter, n_results)
    
    def get_search_explanation(self, results: List[Dict], query: str) -> str:
        """Generate explanation of how search results were determined"""
        if not results:
            return "No relevant sermon content found for your query."
        
        explanation_parts = []
        
        top_result = results[0]
        if 'search_method' in top_result:
            explanation_parts.append("Used enhanced search with multiple ranking factors.")
        
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
        if avg_confidence > 0.8:
            explanation_parts.append("High confidence matches found.")
        elif avg_confidence > 0.6:
            explanation_parts.append("Good matches found with moderate confidence.")
        else:
            explanation_parts.append("Matches found but with lower confidence - you might want to rephrase your question.")
        
        factors_used = []
        if any('combined_score' in r for r in results):
            factors_used.append("semantic similarity and keyword matching")
        if any('relevance_score' in r for r in results):
            factors_used.append("content relevance analysis")
        if any('context_added' in r for r in results):
            factors_used.append("surrounding context")
        if any('detected_topics' in r for r in results):
            factors_used.append("topic categorization")
        
        if factors_used:
            explanation_parts.append(f"Ranking based on: {', '.join(factors_used)}.")
        
        pastors = list(set(r['metadata'].get('pastor', 'Unknown') for r in results[:3]))
        if len(pastors) == 1:
            explanation_parts.append(f"Results primarily from {pastors[0]}.")
        elif len(pastors) <= 3:
            explanation_parts.append(f"Results from pastors: {', '.join(pastors)}.")
        
        return " ".join(explanation_parts)
    
    def configure_search_options(self, 
                               enable_hybrid: bool = True,
                               enable_reranking: bool = True, 
                               enable_context: bool = True,
                               context_window: int = 1,
                               enable_metadata_weighting: bool = True) -> Dict:
        """Configure search enhancement options"""
        
        options = {
            'use_hybrid': enable_hybrid and SEARCH_ENHANCEMENTS,
            'use_reranking': enable_reranking and SEARCH_ENHANCEMENTS,
            'expand_context': enable_context and SEARCH_ENHANCEMENTS,
            'context_window': context_window,
            'max_context_expansions': 3,
            'use_metadata_weighting': enable_metadata_weighting and SEARCH_ENHANCEMENTS,
            'user_preferences': getattr(st.session_state, 'user_search_preferences', None)
        }
        
        st.session_state.search_options = options
        
        return options
    
    def update_user_search_preferences(self, preferences: Dict):
        """Update user's search preferences"""
        st.session_state.user_search_preferences = preferences
        
        if ENHANCED_FEATURES:
            logger.log_user_activity("search_preferences_updated", {
                'favorite_pastors': len(preferences.get('favorite_pastors', [])),
                'preferred_topics': len(preferences.get('preferred_topics', [])),
                'avoid_topics': len(preferences.get('avoid_topics', [])),
                'prefer_recent': preferences.get('prefer_recent', False)
            })
    
    @handle_errors(context="ai_response_generation", user_message="AI response temporarily unavailable")
    def generate_conversational_answer(self, query: str, context_results: List[Dict], conversation_context: str = "", use_grok: bool = False) -> str:
        """Generate conversational answer using AI model with chat context"""
        if not context_results:
            if ENHANCED_FEATURES:
                logger.log_user_activity("no_results_found", {"query_length": len(query)})
            return "I don't see anything in your sermon library that addresses that topic. Could you try asking about something else, or perhaps rephrase your question?"
        
        # Use expanded content if available, otherwise use original content
        context_text = "\n\n".join([
            f"From {result['metadata']['pastor']} in \"{result['metadata'].get('title', result['metadata']['filename'])}\":\n{result.get('expanded_content', result['content'])}"
            for result in context_results
        ])
        
        system_prompt = """You are a knowledgeable assistant helping someone analyze their sermon library. Provide clear, direct responses focused on the sermon content.

Key guidelines:
- Start immediately with your answer - no introductory paragraphs or acknowledgments
- Reference pastors by name and sermon title when discussing their teachings (e.g., "Pastor John taught in 'Walking in Faith' that...")
- Present information objectively and clearly
- Make connections between different sermons when relevant
- Focus on theological content and biblical insights
- Provide substantive analysis rather than commentary about the request
- When appropriate, note related topics that might be worth exploring
- Never acknowledge the format or structure of the request - just answer directly"""
        
        context_addition = f"\n\nPrevious conversation context:\n{conversation_context}" if conversation_context else ""
        
        user_prompt = f"""Based on our conversation, you asked: "{query}"

Here's what I found in your sermon library:
{context_text}{context_addition}

Please respond in a natural, conversational way."""

        if use_grok and st.session_state.get('grok_api_key'):
            return self.generate_grok_response(system_prompt, user_prompt)
        else:
            return self.generate_local_response(system_prompt, user_prompt)
    
    @log_ai_operation_decorator("grok")
    def generate_grok_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Grok (xAI) API"""
        try:
            client = OpenAI(
                api_key=st.session_state.get('grok_api_key'),
                base_url="https://api.x.ai/v1"
            )
            
            response = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            
            result = response.choices[0].message.content
            
            if ENHANCED_FEATURES:
                logger.log_ai_operation("grok_response", "grok", True, details={
                    'prompt_length': len(user_prompt),
                    'response_length': len(result)
                })
            
            return result
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_ai_operation("grok_response", "grok", False, details={'error': str(e)})
                raise AIModelError(f"Error generating Grok response: {e}")
            else:
                return f"Error generating Grok response: {e}"
    
    @log_ai_operation_decorator("ollama")
    def generate_local_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using local Ollama model or provide fallback"""
        try:
            ollama_url = "http://localhost:11434/api/generate"
            models_to_try = ["llama3.2", "llama2", "mistral", "codellama"]
            
            for model in models_to_try:
                try:
                    payload = {
                        "model": model,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_ctx": 4096
                        }
                    }
                    
                    response = requests.post(ollama_url, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'response' in result and result['response'].strip():
                            if ENHANCED_FEATURES:
                                logger.log_ai_operation("ollama_response", model, True, details={
                                    'model_used': model,
                                    'response_length': len(result['response'])
                                })
                            return result['response']
                    elif response.status_code == 404:
                        continue
                        
                except requests.exceptions.RequestException as e:
                    if ENHANCED_FEATURES:
                        logger.log_ai_operation("ollama_response", model, False, details={
                            'model_attempted': model,
                            'error': str(e)
                        })
                    continue
            
            if ENHANCED_FEATURES:
                logger.log_ai_operation("ollama_response", "all_models", False, details={
                    'models_tried': models_to_try,
                    'fallback_used': True
                })
            return self.generate_fallback_response(user_prompt)
                
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_ai_operation("ollama_response", "unknown", False, details={'error': str(e)})
            return self.generate_fallback_response(user_prompt)
    
    def generate_fallback_response(self, user_prompt: str) -> str:
        """Provide a basic response when no AI model is available"""
        if ENHANCED_FEATURES:
            logger.log_app_event("fallback_response_used", {"prompt_length": len(user_prompt)}, level="warning")
        return """I found relevant sermon content, but encountered an issue with the AI models.

**Troubleshooting Ollama:**
1. Check if Ollama is running: `ollama list` 
2. Try downloading a model: `ollama pull llama3.2`
3. Start Ollama service: `ollama serve`

**Or switch to Grok AI:**
- Go to Settings tab and enable Grok AI 

Please review the "Relevant Sermon Excerpts" below for the information related to your question."""

    @log_db_operation_decorator("remove_sermon")
    def remove_sermon(self, filename: str) -> bool:
        """Remove specific sermon from the database"""
        try:
            results = self.collection.get(where={"filename": filename})
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                if ENHANCED_FEATURES:
                    logger.log_db_operation("remove_sermon", success=True, details={
                        "filename": filename,
                        "chunks_removed": len(results['ids'])
                    })
                return True
            
            if ENHANCED_FEATURES:
                logger.log_db_operation("remove_sermon", success=False, details={
                    "filename": filename,
                    "reason": "not_found"
                })
            return False
            
        except Exception as e:
            if ENHANCED_FEATURES:
                logger.log_db_operation("remove_sermon", success=False, details={
                    "filename": filename,
                    "error": str(e)
                })
            st.error(f"Error removing sermon: {e}")
            return False

    @handle_errors(context="database_clear", user_message="Database clearing failed")
    def clear_database(self):
        """Clear all sermon data from the database"""
        try:
            db_path = Path("./sermon_db")
            if db_path.exists():
                import os
                
                if not os.access(db_path, os.W_OK):
                    if ENHANCED_FEATURES:
                        logger.log_db_operation("clear_database", success=False, details={
                            "reason": "readonly_directory"
                        })
                    return False, "Database directory is read-only. Check file permissions."
                
                for file_path in db_path.rglob('*'):
                    if file_path.is_file() and not os.access(file_path, os.W_OK):
                        if ENHANCED_FEATURES:
                            logger.log_db_operation("clear_database", success=False, details={
                                "reason": "readonly_file",
                                "file": file_path.name
                            })
                        return False, f"Database file is read-only: {file_path.name}. Check file permissions."
            
            self.client.delete_collection("sermons")
            
            self.collection = self.client.get_or_create_collection(
                name="sermons",
                metadata={"description": "Sermon transcripts and chunks"}
            )
            
            # Clear caches
            self._query_cache.clear()
            self._keyword_index.clear()
            self._document_texts.clear()
            
            if ENHANCED_FEATURES:
                logger.log_db_operation("clear_database", success=True, details={
                    "operation": "collection_recreated"
                })
            
            return True, "Database cleared successfully"
            
        except Exception as e:
            error_msg = str(e)
            if ENHANCED_FEATURES:
                logger.log_db_operation("clear_database", success=False, details={
                    "error": error_msg
                })
            
            if "readonly database" in error_msg.lower():
                return False, "Database is read-only. Try restarting the application or check file permissions."
            elif "permission" in error_msg.lower():
                return False, "Permission denied. Check file permissions on the sermon_db directory."
            else:
                return False, f"Error clearing database: {error_msg}"