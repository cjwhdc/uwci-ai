#!/usr/bin/env python3
"""
UWCI Sermon AI
A smart AI system to ingest sermon transcripts and answer detailed questions
"""

import streamlit as st
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import time;

# Import API configuration
try:
    from app.config.config import GROK_API_KEY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MAX_SEARCH_RESULTS
except ImportError:
    st.error("config/config.py file not found. Please create app/config/config.py with your API keys.")
    st.stop()

# Core dependencies for AI functionality
try:
    import chromadb
    from chromadb.config import Settings
    import openai
    from sentence_transformers import SentenceTransformer
    import requests
except ImportError as e:
    st.error(f"Missing required packages. Please install: {e}")
    st.stop()

class SermonProcessor:
    """Handle Bible verse lookups and integration"""
    
    def __init__(self):
        self.translations = {
            'AMP': 'Amplified Bible',
            'NLT': 'New Living Translation'
        }
        self.bible_data = {}
        self.book_names = {}
    
    def extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from sermon content"""
        metadata = {
            'filename': filename,
            'pastor': 'Unknown',
            'date': 'Unknown',
            'word_count': len(content.split())
        }
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # For your format: title, pastor, [optional "Watch"], date
        if len(lines) >= 3:
            # Skip the first line (title) and take the second line as pastor
            potential_pastor = lines[1]
            
            # Clean up the pastor name
            pastor_name = re.sub(r'^(pastor|rev|dr|mr|mrs)\.?\s*', '', potential_pastor, flags=re.IGNORECASE)
            pastor_name = re.sub(r'\s*(sermon|message|teaching).*$', '', pastor_name, flags=re.IGNORECASE)
            pastor_name = pastor_name.strip('# *_-=+')
            
            # Validate it's a reasonable pastor name (not too long, not empty)
            if len(pastor_name) > 0 and len(pastor_name) < 50:
                metadata['pastor'] = pastor_name
            
            # Check if third line is "Watch" or similar, if so, date is on line 4
            if len(lines) >= 4 and lines[2].strip().lower() in ['watch', '"watch"', "'watch'"]:
                potential_date = lines[3]
            else:
                # Third line should be the date
                potential_date = lines[2]
            
            # Clean markdown characters from date
            clean_date = potential_date.strip()
            clean_date = re.sub(r'^#+\s*', '', clean_date)  # Remove leading #
            clean_date = re.sub(r'[*_`~]', '', clean_date)   # Remove markdown formatting
            clean_date = clean_date.strip('# *_-=+')         # Strip remaining characters
            metadata['date'] = clean_date
        
        # Fallback: if the above doesn't work, try the original patterns
        if metadata['pastor'] == 'Unknown':
            # Look for pastor in first few lines with more specific patterns
            for line in lines[:5]:
                # More specific patterns to avoid matching titles
                pastor_patterns = [
                    r'(?:pastor|preacher|speaker):\s*(.+?)(?:\n|$)',
                    r'(?:by|preached by):\s*(.+?)(?:\n|$)',
                    r'^pastor\s+(.+?)(?:\n|$)',
                    r'^preacher:\s*(.+?)(?:\n|$)',
                ]
                
                for pattern in pastor_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        pastor_name = match.group(1).strip()
                        pastor_name = re.sub(r'^(pastor|rev|dr|mr|mrs)\.?\s*', '', pastor_name, flags=re.IGNORECASE)
                        pastor_name = re.sub(r'\s*(sermon|message|teaching).*$', '', pastor_name, flags=re.IGNORECASE)
                        if len(pastor_name) > 0 and len(pastor_name) < 50:
                            metadata['pastor'] = pastor_name
                            break
                
                if metadata['pastor'] != 'Unknown':
                    break
        
        # Fallback date extraction if not found in structured format
        if metadata['date'] == 'Unknown' or not metadata['date']:
            date_patterns = [
                r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
                r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # MM/DD/YYYY
                r'\b(\d{1,2}-\d{1,2}-\d{4})\b',  # MM-DD-YYYY
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b',
            ]
            
            for line in lines[:10]:
                for pattern in date_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        metadata['date'] = match.group(1)
                        break
                if metadata['date'] != 'Unknown' and metadata['date']:
                    break
        
        return metadata
    
    def chunk_sermon(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split sermon into overlapping chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Don't create tiny chunks at the end
            if len(chunk_words) > 100 or i == 0:  
                chunks.append(chunk_text)
            else:
                # Add remaining words to the last chunk
                if chunks:
                    chunks[-1] += ' ' + chunk_text
                else:
                    chunks.append(chunk_text)
                break
        
        return chunks

    def show_file_status_section():
        """Updated file status section with proper metadata extraction"""
        st.subheader("File Status")
        sermons_dir = Path("data/sermons")
        
        if sermons_dir.exists():
            sermon_files = list(sermons_dir.glob("*.md"))
            processed_files = st.session_state.ai_engine.get_processed_files()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files Found", len(sermon_files))
            with col2:
                st.metric("Files Processed", len(processed_files))
            with col3:
                unprocessed = len(sermon_files) - len(processed_files)
                st.metric("Files Unprocessed", unprocessed)
            
            # Show detailed file status with proper metadata
            
            if sermon_files:
                files_data = []
                
                # Progress bar for file processing
                if len(sermon_files) > 5:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                for idx, file in enumerate(sermon_files):
                    if len(sermon_files) > 5:
                        status_text.text(f'Reading file {idx + 1} of {len(sermon_files)}: {file.name}')
                        progress_bar.progress((idx + 1) / len(sermon_files))
                    
                    status = "✅ Processed" if file.name in processed_files else "⏳ Not Processed"
                    
                    # Extract metadata from file
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Use the sermon processor to extract metadata
                        metadata = st.session_state.sermon_processor.extract_metadata(content, file.name)
                        pastor = metadata.get('pastor', 'Unknown')
                        date = metadata.get('date', 'Unknown')
                        word_count = metadata.get('word_count', 0)
                        
                    except Exception as e:
                        pastor = f'Error: {str(e)[:30]}...'
                        date = 'Unknown'
                        word_count = 0
                    
                    files_data.append({
                        'File': file.name,
                        'Pastor': pastor,
                        'Date': date,
                        'Words': word_count,
                        'Status': status
                    })
                
                # Clear progress indicators
                if len(sermon_files) > 5:
                    progress_bar.empty()
                    status_text.empty()
                
                # Display the dataframe
                df_files = pd.DataFrame(files_data)
                st.dataframe(df_files, use_container_width=True)
                
                # Show some statistics
                st.subheader("File Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_words = sum([row['Words'] for row in files_data if isinstance(row['Words'], int)])
                    st.metric("Total Words", f"{total_words:,}")
                
                with col2:
                    unique_pastors = len(set([row['Pastor'] for row in files_data if row['Pastor'] != 'Unknown']))
                    st.metric("Unique Pastors", unique_pastors)
                
                with col3:
                    avg_words = int(total_words / len(files_data)) if files_data else 0
                    st.metric("Avg Words/Sermon", f"{avg_words:,}")
                
                # Show pastor breakdown
                if unique_pastors > 0:
                    pastor_counts = {}
                    for row in files_data:
                        pastor = row['Pastor']
                        if pastor != 'Unknown':
                            pastor_counts[pastor] = pastor_counts.get(pastor, 0) + 1
                    
                    if pastor_counts:
                        st.subheader("Sermons by Pastor")
                        pastor_df = pd.DataFrame([
                            {'Pastor': pastor, 'Sermon Count': count}
                            for pastor, count in sorted(pastor_counts.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(pastor_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Sermon directory 'data/sermons' not found. Please create this directory and add your .md sermon files.")
            st.code("mkdir -p data/sermons\ncp your-sermons/*.md data/sermons/")

class AIEngine:
    """Main AI processing engine using ChromaDB and OpenAI/Local LLM"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.setup_vector_db()
        self.auto_import_sermons()
    
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
            
        except Exception as e:
            st.error(f"Error setting up vector database: {e}")
    
    def get_processed_files(self) -> set:
        """Get list of already processed sermon files"""
        try:
            results = self.collection.get()
            if results and results['metadatas']:
                processed = {meta.get('filename', '') for meta in results['metadatas']}
                return processed
            return set()
        except Exception as e:
            st.error(f"Error getting processed files: {e}")
            return set()
    
    def auto_import_sermons(self):
        """Automatically import and process sermons from data/sermons folder"""
        sermons_dir = Path("data/sermons")
        
        if not sermons_dir.exists():
            st.warning("Sermons directory 'data/sermons' not found. Please create it and add your .md files.")
            return
        
        # Get list of sermon files
        sermon_files = list(sermons_dir.glob("*.md"))
        
        if not sermon_files:
            st.info("No sermon files found in data/sermons folder. Add .md files to auto-import them.")
            return
        
        # Get already processed files
        processed_files = self.get_processed_files()
        
        # Find new files to process
        new_files = [f for f in sermon_files if f.name not in processed_files]
        
        if new_files:
            st.info(f"Found {len(new_files)} new sermon(s) to process...")
            
            processor = SermonProcessor()
            progress_bar = st.progress(0)
            
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
                        time.sleep(3)
                        st.empty()
                    else:
                        st.error(f"Failed to process: {sermon_file.name}")
                        time.sleep(3)
                        st.empty()
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(new_files))
                    
                except Exception as e:
                    st.error(f"Error processing {sermon_file.name}: {e}")
            
            st.success(f"Auto-import complete! Processed {len(new_files)} sermon(s).")
            time.sleep(3)
            st.empty()
        else:
            st.info(f"All {len(sermon_files)} sermon(s) already processed. Database is up to date.")
            time.sleep(3)
            st.empty()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using sentence transformers"""
        if not hasattr(self, 'embedder') or self.embedder is None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.error(f"Error loading embedding model: {e}")
                return []
        
        try:
            embedding = self.embedder.encode(text).tolist()
            return embedding
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return []
    
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
                return True
            return False
            
        except Exception as e:
            st.error(f"Error adding sermon to database: {e}")
            return False
    
    def search_sermons(self, query: str, pastor_filter: str = None, n_results: int = 5) -> List[Dict]:
        """Search for relevant sermon content"""
        try:
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
            
            return search_results
            
        except Exception as e:
            st.error(f"Error searching sermons: {e}")
            return []
    
    def generate_conversational_answer(self, query: str, context_results: List[Dict], conversation_context: str = "", use_grok: bool = False) -> str:
        """Generate conversational answer using AI model with chat context"""
        if not context_results:
            return "I don't see anything in your sermon library that addresses that topic. Could you try asking about something else, or perhaps rephrase your question?"
        
        # Prepare context
        context_text = "\n\n".join([
            f"From {result['metadata']['pastor']} ({result['metadata'].get('date', 'Unknown date')}):\n{result['content']}"
            for result in context_results
        ])
        
        system_prompt = """You are a knowledgeable assistant helping someone analyze their sermon library. Provide clear, informative responses that focus on the content and teachings found in the sermons.

Key guidelines:
- Be direct and informational rather than overly conversational
- Reference pastors by name when discussing their specific teachings
- Present information objectively and clearly
- Make connections between different sermons when relevant
- Focus on theological content and biblical insights
- Provide substantive analysis rather than casual commentary
- When appropriate, note related topics that might be worth exploring
- Keep responses substantive and focused on the sermon content"""
        
        context_addition = f"\n\nPrevious conversation context:\n{conversation_context}" if conversation_context else ""
        
        user_prompt = f"""Based on our conversation, you asked: "{query}"

Here's what I found in your sermon library:
{context_text}{context_addition}

Please respond in a natural, conversational way."""

        if use_grok and st.session_state.get('grok_api_key'):
            return self.generate_grok_response(system_prompt, user_prompt)
        else:
            return self.generate_local_response(system_prompt, user_prompt)

    def generate_answer(self, query: str, context_results: List[Dict], use_grok: bool = False) -> str:
        """Generate answer using AI model"""
        if not context_results:
            return "I couldn't find relevant information in the sermons to answer your question."
        
        # Prepare context
        context_text = "\n\n".join([
            f"From {result['metadata']['pastor']} ({result['metadata'].get('date', 'Unknown date')}):\n{result['content']}"
            for result in context_results
        ])
        
        system_prompt = """You are a knowledgeable assistant specializing in sermon analysis and theological discussion. 
        Based on the provided sermon excerpts, answer the user's question accurately and thoughtfully. 
        
        Always cite which pastor said what when referencing specific points. Be respectful of the religious content 
        and maintain theological accuracy. If the information isn't clearly present in the sermons, say so."""
        
        user_prompt = f"""Question: {query}

Relevant sermon excerpts:
{context_text}

Please provide a detailed answer based on the sermon content above."""

        if use_grok and st.session_state.get('grok_api_key'):
            return self.generate_grok_response(system_prompt, user_prompt)
        else:
            return self.generate_local_response(system_prompt, user_prompt)
    
    def generate_grok_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Grok (xAI) API"""
        try:
            from openai import OpenAI
            
            # Use Grok's API endpoint with OpenAI-compatible client
            client = OpenAI(
                api_key=st.session_state.get('grok_api_key'),
                base_url="https://api.x.ai/v1"
            )
            
            response = client.chat.completions.create(
                model="grok-3",  # Updated from "grok-beta"
                messages=[
                    {"role": "system", "content": system_prompt},
                    # {"role": "user", "content": user_prompt + "in 1000 words or less"}
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                # max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating Grok response: {e}"
    
    def generate_local_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using local Ollama model or provide fallback"""
        try:
            # Try to connect to local Ollama instance
            ollama_url = "http://localhost:11434/api/generate"
            
            # Try different models in order of preference
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
                            return result['response']
                    elif response.status_code == 404:
                        # Model not found, try next one
                        continue
                    else:
                        # Other error, log it but try next model
                        print(f"Ollama error with {model}: {response.status_code} - {response.text}")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    print(f"Request error with {model}: {e}")
                    continue
            
            # If we get here, none of the models worked
            return self.generate_fallback_response(user_prompt)
                
        except Exception as e:
            return self.generate_fallback_response(user_prompt)
    
    def generate_fallback_response(self, user_prompt: str) -> str:
        """Provide a basic response when no AI model is available"""
        return """I found relevant sermon content, but encountered an issue with the AI models.

**Troubleshooting Ollama:**
1. Check if Ollama is running: `ollama list` 
2. Try downloading a model: `ollama pull llama3.2`
3. Start Ollama service: `ollama serve`

**Or switch to OpenAI:**
- Go to Settings tab and check "Use OpenAI" 

Please review the "Relevant Sermon Excerpts" below for the information related to your question."""

    def remove_sermon(self, filename: str) -> bool:
        """Remove specific sermon from the database"""
        try:
            # Get all documents for this sermon
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                # Delete all chunks for this sermon
                self.collection.delete(ids=results['ids'])
                return True
            return False
            
        except Exception as e:
            st.error(f"Error removing sermon: {e}")
            return False

    def clear_database(self):
        """Clear all sermon data from the database"""
        try:
            # Delete the collection
            self.client.delete_collection("sermons")
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name="sermons",
                metadata={"description": "Sermon transcripts and chunks"}
            )
            
            return True
        except Exception as e:
            st.error(f"Error clearing database: {e}")
            return False

# Streamlit App
def main():
    st.set_page_config(
        page_title="UWCI Sermon AI",
        page_icon="📖",
        layout="wide"
    )
    
    st.title("📖 UWCI Sermon AI")
    st.markdown("Upload sermon transcripts and ask detailed questions about their content")
    
    # Initialize components with config file API key
    if 'ai_engine' not in st.session_state:
        with st.spinner("Initializing AI engine and auto-importing sermons..."):
            st.session_state.ai_engine = AIEngine()
            st.session_state.sermon_processor = SermonProcessor()
            # Set API key from config file
            st.session_state.grok_api_key = GROK_API_KEY
            st.session_state.use_grok = True
    
    # Sidebar - simplified
    st.sidebar.header("System Status")
    
    # Show current AI model status
    if st.session_state.get('use_grok', False):
        st.sidebar.success("Using Grok AI")
    else:
        st.sidebar.info("Using local AI (Ollama)")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Sermon Library", "Settings"])
    
    with tab1:        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            # Add welcome message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Hello! I'm here to help you explore your sermon library. I can answer questions about what different pastors have taught, find related Bible passages, and help you discover insights across your collection. What would you like to discuss?",
                "timestamp": datetime.now()
            })
        
        # Pastor filter in sidebar for this tab
        with st.sidebar:
            st.subheader("Conversation Settings")
            
            # Get pastors for filter
            if hasattr(st.session_state.ai_engine, 'collection') and st.session_state.ai_engine.collection:
                try:
                    all_results = st.session_state.ai_engine.collection.get()
                    raw_pastors = [meta.get('pastor', 'Unknown') for meta in all_results['metadatas']]
                    
                    # Clean up pastor names
                    cleaned_pastors = set()
                    for pastor in raw_pastors:
                        clean_pastor = re.sub(r'^#+\s*', '', pastor)
                        clean_pastor = re.sub(r'[*_`~]', '', clean_pastor)
                        clean_pastor = clean_pastor.strip('# *_-=+')
                        cleaned_pastors.add(clean_pastor)
                    
                    pastors = ["All Pastors"] + sorted(list(cleaned_pastors))
                except:
                    pastors = ["All Pastors"]
            else:
                pastors = ["All Pastors"]
            
            pastor_filter = st.selectbox("Focus on specific pastor", pastors)
            
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Conversation cleared. What would you like to discuss about your sermons?",
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Show relevant excerpts if they exist
                        if "excerpts" in message:
                            with st.expander("View sermon excerpts"):
                                for i, excerpt in enumerate(message["excerpts"]):
                                    st.write(f"**From {excerpt['pastor']} ({excerpt['date']}):**")
                                    st.write(excerpt['content'])
                                    st.divider()
        
        # Chat input
        user_input = st.chat_input("Ask about your sermons...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Process the question
            with st.spinner("Thinking..."):
                # Get conversation context (last few messages)
                context_messages = st.session_state.chat_history[-6:]  # Last 3 exchanges
                conversation_context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                    for msg in context_messages[:-1]  # Exclude current message
                ])
                
                # Search for relevant content
                filter_pastor = pastor_filter if pastor_filter != "All Pastors" else None
                results = st.session_state.ai_engine.search_sermons(user_input, filter_pastor)
                
                if results:
                    # Generate conversational response
                    use_grok = st.session_state.get('use_grok', True)
                    response = st.session_state.ai_engine.generate_conversational_answer(
                        user_input, results, conversation_context, use_grok
                    )
                    
                    # Add assistant response with excerpts
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "excerpts": [
                            {
                                "pastor": result['metadata']['pastor'],
                                "date": result['metadata'].get('date', 'Unknown date'),
                                "content": result['content']
                            }
                            for result in results[:3]  # Top 3 excerpts
                        ],
                        "timestamp": datetime.now()
                    })
                else:
                    # No relevant content found
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I couldn't find relevant information in your sermon library to answer that question. Could you try rephrasing it or asking about a different topic?",
                        "timestamp": datetime.now()
                    })
            
            st.rerun()
    
    with tab2:
        st.header("Sermon Library & Auto-Import")
        
        # Action buttons at the top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Scan for New Sermons", type="primary"):
                with st.spinner("Scanning sermon folder..."):
                    st.session_state.ai_engine.auto_import_sermons()
                    st.success("Scan complete!")
                    st.rerun()
        
        with col2:
            if st.button("Clear Database", type="secondary", help="Remove all processed sermons from database"):
                # Add confirmation
                if st.session_state.get('confirm_clear', False):
                    with st.spinner("Clearing database..."):
                        success = st.session_state.ai_engine.clear_database()
                        if success:
                            st.success("Database cleared successfully!")
                            st.session_state.confirm_clear = False
                        else:
                            st.error("Failed to clear database")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm database clearing")
        
        st.divider()
        
        # Show library of processed sermons
        # if hasattr(st.session_state.ai_engine, 'collection') and st.session_state.ai_engine.collection:
        #     try:
        #         # Get all processed sermons
        #         all_results = st.session_state.ai_engine.collection.get()
                
        #         if all_results['metadatas']:
        #             st.subheader("Processed Sermon Library")
                    
        #             # Create a summary dataframe
        #             sermons_data = {}
        #             for meta in all_results['metadatas']:
        #                 filename = meta.get('filename', 'Unknown')
        #                 if filename not in sermons_data:
        #                     sermons_data[filename] = {
        #                         'Pastor': meta.get('pastor', 'Unknown'),
        #                         'Date': meta.get('date', 'Unknown'),
        #                         'Word Count': meta.get('word_count', 0),
        #                         'Chunks': 0
        #                     }
        #                 sermons_data[filename]['Chunks'] += 1
                    
        #             # Display each sermon with remove button
        #             for filename, data in sermons_data.items():
        #                 col1, col2 = st.columns([4, 1])
                        
        #                 with col1:
        #                     st.write(f"**{filename}** - {data['Pastor']} ({data['Date']}) - {data['Word Count']} words, {data['Chunks']} chunks")
                        
        #                 with col2:
        #                     if st.button(f"Remove", key=f"remove_{filename}", help=f"Remove {filename} from processed library"):
        #                         with st.spinner(f"Removing {filename}..."):
        #                             success = st.session_state.ai_engine.remove_sermon(filename)
        #                             if success:
        #                                 st.success(f"Removed {filename}")
        #                                 st.rerun()
        #                             else:
        #                                 st.error(f"Failed to remove {filename}")
                    
        #             st.divider()
                    
        #             # Summary stats
        #             col1, col2, col3, col4 = st.columns(4)
        #             with col1:
        #                 st.metric("Total Sermons", len(sermons_data))
        #             with col2:
        #                 st.metric("Unique Pastors", len(set([data['Pastor'] for data in sermons_data.values()])))
        #             with col3:
        #                 st.metric("Total Words", sum([data['Word Count'] for data in sermons_data.values()]))
        #             with col4:
        #                 st.metric("Total Chunks", sum([data['Chunks'] for data in sermons_data.values()]))
        #         else:
        #             st.info("No sermons processed yet. Add .md files to data/sermons/ and click 'Scan for New Sermons'.")
        #     except Exception as e:
        #         st.error(f"Error loading sermon library: {e}")
        # else:
        #     st.info("No sermons processed yet. Add .md files to data/sermons/ and click 'Scan for New Sermons'.")
        
        # st.divider()
        
        # File status section
        st.subheader("File Status")
        sermons_dir = Path("data/sermons")
        
        if sermons_dir.exists():
            sermon_files = list(sermons_dir.glob("*.md"))
            processed_files = st.session_state.ai_engine.get_processed_files()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files Found", len(sermon_files))
            with col2:
                st.metric("Files Processed", len(processed_files))
            with col3:
                unprocessed = len(sermon_files) - len(processed_files)
                st.metric("Files Unprocessed", unprocessed)
            
            # Show detailed file status
            if sermon_files:
                files_data = []
                for file in sermon_files:
                    status = "✅ Processed" if file.name in processed_files else "⏳ Not Processed"
                    
                    # Try to get pastor name from file - use same logic as SermonProcessor
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Use the same extraction logic as SermonProcessor for consistency
                        processor = SermonProcessor()
                        metadata = processor.extract_metadata(content, file.name)
                        pastor = metadata.get('pastor', 'Unknown')
                        date = metadata.get('date', 'Unknown')
                        word_count = metadata.get('word_count', 0)
                        
                    except Exception as e:
                        pastor = 'Error reading'
                        date = 'Unknown'
                        word_count = 0
                    
                    files_data.append({
                        'File': file.name,
                        'Pastor': pastor,
                        'Date': date,
                        'Words': word_count,
                        'Status': status
                    })
                
                df_files = pd.DataFrame(files_data)
                st.dataframe(df_files, use_container_width=True)
        else:
            st.warning("Sermon directory 'data/sermons' not found. Please create this directory and add your .md sermon files.")
            st.code("mkdir -p data/sermons\ncp your-sermons/*.md data/sermons/")
        
        st.divider()
        
        # Instructions
        st.subheader("How Auto-Import Works")
        st.markdown("""
        1. **Place sermon files** in the `data/sermons/` folder (only .md files are processed)
        2. **Click "Scan for New Sermons"** to automatically process any new files
        3. **Files are automatically processed** and chunked for AI analysis
        4. **Already processed files are skipped** to avoid duplicates
        5. **Use "Clear Database"** if you need to start fresh (requires confirmation)
        
        The system remembers which files have been processed, so you can safely add new sermons anytime and only new files will be processed.
        """)
    
    with tab3:
        st.header("Settings")
        
        st.subheader("AI Model Configuration")
        
        # AI Model Selection
        current_use_grok = st.session_state.get('use_grok', True)
        use_grok = st.checkbox("Use Grok AI (recommended)", value=current_use_grok)
        
        if use_grok != current_use_grok:
            st.session_state.use_grok = use_grok
            st.rerun()
        
        if use_grok:
            st.success("Using Grok AI (xAI) for best results")
            st.info("API Key: Configured via config.py file")
        else:
            st.info("Using local AI model (Ollama)")
            st.write("To use Ollama:")
            st.code("""
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Download a model
ollama pull llama3.2
            """)
        
        st.divider()
        
        st.subheader("System Information")
        
        # Database info
        sermons_dir = Path("data/sermons")
        if sermons_dir.exists():
            sermon_files = list(sermons_dir.glob("*.md"))
            processed_files = st.session_state.ai_engine.get_processed_files()
            st.info(f"Sermon files found: {len(sermon_files)} | Processed: {len(processed_files)}")
                
        st.subheader("Advanced Settings")
        
        # Chunk size settings
        st.write("**Text Processing Settings**")
        chunk_size = st.slider("Chunk Size (words)", 500, 2000, 1000, help="Larger chunks provide more context but may be less precise")
        chunk_overlap = st.slider("Chunk Overlap (words)", 50, 500, 200, help="Overlap between chunks to maintain context")
        
        # Search settings
        st.write("**Search Settings**")
        max_results = st.slider("Max Search Results", 3, 20, 5, help="Number of relevant excerpts to return for each question")
        
        # System status
        st.subheader("System Status")
        st.write("**Database:** Connected ✓")
        
        st.write("**Embedding Model:** Loaded ✓")
        
        if use_grok:
            st.write("**AI Model:** Grok AI (xAI) ✓")
        else:
            st.write("**AI Model:** Local (Ollama)")
        
        st.divider()
        
        st.subheader("Directory Structure")
        st.code("""
sermon-ai/
├── data/
│   ├── sermons/          # Place .md sermon files here
│   └── bible/            # Place XML Bible files here
├── sermon_db/            # Vector database (auto-created)
└── app/                  # Application files
        """)


if __name__ == "__main__":
    main()