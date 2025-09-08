import streamlit as st
import time
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Fix for SQLite version issue on Streamlit Cloud (optional)
try:
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # pysqlite3 not available, use system sqlite3
    pass

import chromadb
from sentence_transformers import SentenceTransformer
import requests
from openai import OpenAI

from .sermon_processor import SermonProcessor
from .utils.error_handler import handle_errors, ErrorHandler, AIModelError, DatabaseError
from .utils.logger import logger, log_ai_operation_decorator, log_db_operation_decorator

class AIEngine:
    """Main AI processing engine using ChromaDB and OpenAI/Local LLM with enhanced error handling"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.error_handler = ErrorHandler()
        
        # Query cache for performance
        self._query_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        # Try to sync database from cloud first
        self.sync_database_from_cloud()
        
        self.setup_vector_db()
    
    @handle_errors(context="cloud_sync", user_message="Cloud sync temporarily unavailable")
    def sync_database_from_cloud(self):
        """Sync database from Dropbox if configured"""
        try:
            from app.config.config import USE_DROPBOX_SYNC, DropboxDatabaseSync
            if USE_DROPBOX_SYNC:
                sync = DropboxDatabaseSync()
                if sync.sync_from_dropbox():
                    logger.log_app_event("dropbox_sync_success", {"direction": "download"})
                    return True
        except (ImportError, AttributeError):
            # Dropbox sync not configured, continue with local database
            pass
        except Exception as e:
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
                    logger.log_app_event("dropbox_sync_success", {"direction": "upload"})
                else:
                    logger.log_app_event("dropbox_sync_failed", {"direction": "upload"}, level="warning")
                return success
        except (ImportError, AttributeError):
            return False
        except Exception as e:
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
            logger.log_app_event("backup_info_failed", {"error": str(e)}, level="error")
            st.error(f"Error getting backup info: {e}")
        return None
    
    @handle_errors(context="vector_db_setup", user_message="Database initialization failed")
    def setup_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(path="./sermon_db")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="sermons",
                metadata={"description": "Sermon transcripts and chunks"}
            )
            
            logger.log_db_operation("database_init", success=True, details={
                "client_type": "persistent",
                "collection_name": "sermons"
            })
            
        except Exception as e:
            logger.log_db_operation("database_init", success=False, details={"error": str(e)})
            raise DatabaseError(f"Error setting up vector database: {e}")
    
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
            logger.log_db_operation("get_processed_files", success=False, details={"error": str(e)})
            st.error(f"Error getting processed files: {e}")
            return set()
    
    @handle_errors(context="sermon_processing", user_message="Sermon processing temporarily unavailable")
    def process_new_sermons(self):
        """Process new sermons from data/sermons folder (Manual processing only)"""
        sermons_dir = Path("data/sermons")
        
        if not sermons_dir.exists():
            st.warning("Sermons directory 'data/sermons' not found. Please create it and add your .md files.")
            return
        
        # Get list of sermon files
        sermon_files = list(sermons_dir.glob("*.md"))
        
        if not sermon_files:
            st.info("No sermon files found in data/sermons folder. Add .md files to process them.")
            return
        
        # Get already processed files
        processed_files = self.get_processed_files()
        
        # Find new files to process
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
                    
                    # Read file content
                    with open(sermon_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Process sermon
                    metadata = processor.extract_metadata(content, sermon_file.name)
                    chunks = processor.chunk_sermon(content)
                    
                    # Add to database
                    success = self.add_sermon(content, metadata, chunks)
                    
                    if success:
                        st.success(f"Successfully processed: {sermon_file.name}")
                        processed_count += 1
                        logger.log_app_event("sermon_processed", {
                            "filename": sermon_file.name,
                            "chunks": len(chunks),
                            "word_count": metadata.get('word_count', 0)
                        })
                    else:
                        st.error(f"Failed to process: {sermon_file.name}")
                        failed_count += 1
                    
                    time.sleep(1)  # Brief pause for UI feedback
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(new_files))
                    
                except Exception as e:
                    st.error(f"Error processing {sermon_file.name}: {e}")
                    failed_count += 1
                    logger.log_app_event("sermon_processing_failed", {
                        "filename": sermon_file.name,
                        "error": str(e)
                    }, level="error")
            
            # Summary message
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
                logger.log_app_event("embedding_model_loaded", {"model": "all-MiniLM-L6-v2"})
            except Exception as e:
                logger.log_app_event("embedding_model_failed", {"error": str(e)}, level="error")
                st.error(f"Error loading embedding model: {e}")
                return []
        
        try:
            embedding = self.embedder.encode(text).tolist()
            return embedding
        except Exception as e:
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
                
                logger.log_db_operation("add_sermon", success=True, details={
                    "filename": metadata['filename'],
                    "chunks_added": len(documents)
                })
                return True
            return False
            
        except Exception as e:
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
        """Search for relevant sermon content with caching"""
        try:
            # Log user activity
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
                    logger.log_db_operation("search_cached", success=True, details={
                        'cache_hit': True,
                        'query_hash': cache_key[:8]
                    })
                    return cached_result
            
            # Use session state setting if available, otherwise default
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
            
            # Clean old cache entries (simple cleanup)
            if len(self._query_cache) > 100:  # Limit cache size
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]
            
            logger.log_db_operation("search", success=True, details={
                'results_count': len(search_results),
                'query_hash': cache_key[:8],
                'cached': False
            })
            
            return search_results
            
        except Exception as e:
            logger.log_db_operation("search", success=False, details={
                'error': str(e),
                'query_hash': self._get_cache_key(query, pastor_filter)[:8]
            })
            st.error(f"Error searching sermons: {e}")
            return []
    
    @handle_errors(context="ai_response_generation", user_message="AI response temporarily unavailable")
    def generate_conversational_answer(self, query: str, context_results: List[Dict], conversation_context: str = "", use_grok: bool = False) -> str:
        """Generate conversational answer using AI model with chat context"""
        if not context_results:
            logger.log_user_activity("no_results_found", {"query_length": len(query)})
            return "I don't see anything in your sermon library that addresses that topic. Could you try asking about something else, or perhaps rephrase your question?"
        
        # Prepare context
        context_text = "\n\n".join([
            f"From {result['metadata']['pastor']} in \"{result['metadata'].get('title', result['metadata']['filename'])}\":\n{result['content']}"
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
            
            logger.log_ai_operation("grok_response", "grok", True, details={
                'prompt_length': len(user_prompt),
                'response_length': len(result)
            })
            
            return result
            
        except Exception as e:
            logger.log_ai_operation("grok_response", "grok", False, details={'error': str(e)})
            raise AIModelError(f"Error generating Grok response: {e}")
    
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
                            logger.log_ai_operation("ollama_response", model, True, details={
                                'model_used': model,
                                'response_length': len(result['response'])
                            })
                            return result['response']
                    elif response.status_code == 404:
                        continue
                        
                except requests.exceptions.RequestException as e:
                    logger.log_ai_operation("ollama_response", model, False, details={
                        'model_attempted': model,
                        'error': str(e)
                    })
                    continue
            
            logger.log_ai_operation("ollama_response", "all_models", False, details={
                'models_tried': models_to_try,
                'fallback_used': True
            })
            return self.generate_fallback_response(user_prompt)
                
        except Exception as e:
            logger.log_ai_operation("ollama_response", "unknown", False, details={'error': str(e)})
            return self.generate_fallback_response(user_prompt)
    
    def generate_fallback_response(self, user_prompt: str) -> str:
        """Provide a basic response when no AI model is available"""
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
                logger.log_db_operation("remove_sermon", success=True, details={
                    "filename": filename,
                    "chunks_removed": len(results['ids'])
                })
                return True
            
            logger.log_db_operation("remove_sermon", success=False, details={
                "filename": filename,
                "reason": "not_found"
            })
            return False
            
        except Exception as e:
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
            # Check if database directory exists and is writable
            db_path = Path("./sermon_db")
            if db_path.exists():
                import os
                import stat
                
                # Check if we have write permissions
                if not os.access(db_path, os.W_OK):
                    logger.log_db_operation("clear_database", success=False, details={
                        "reason": "readonly_directory"
                    })
                    return False, "Database directory is read-only. Check file permissions."
                
                # Check individual files in the database directory
                for file_path in db_path.rglob('*'):
                    if file_path.is_file() and not os.access(file_path, os.W_OK):
                        logger.log_db_operation("clear_database", success=False, details={
                            "reason": "readonly_file",
                            "file": file_path.name
                        })
                        return False, f"Database file is read-only: {file_path.name}. Check file permissions."
            
            # Try to delete and recreate the collection
            self.client.delete_collection("sermons")
            
            self.collection = self.client.get_or_create_collection(
                name="sermons",
                metadata={"description": "Sermon transcripts and chunks"}
            )
            
            # Clear cache
            self._query_cache.clear()
            
            logger.log_db_operation("clear_database", success=True, details={
                "operation": "collection_recreated"
            })
            
            return True, "Database cleared successfully"
            
        except Exception as e:
            error_msg = str(e)
            logger.log_db_operation("clear_database", success=False, details={
                "error": error_msg
            })
            
            if "readonly database" in error_msg.lower():
                return False, "Database is read-only. Try restarting the application or check file permissions."
            elif "permission" in error_msg.lower():
                return False, "Permission denied. Check file permissions on the sermon_db directory."
            else:
                return False, f"Error clearing database: {error_msg}"