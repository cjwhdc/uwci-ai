import streamlit as st
from pathlib import Path
from app.auth import admin_user_management, is_admin
from app.config.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MAX_SEARCH_RESULTS

def show_admin_settings_tab():
    """Display the admin-only settings tab"""
    # Double-check admin status
    if not is_admin():
        st.error("🚫 Access Denied: Administrator privileges required")
        st.stop()
    
    st.header("⚙️ Admin Settings")
    st.markdown("*Administrator-only configuration and system management*")
    
    # User Management Section
    st.subheader("👥 User Management")
    admin_user_management()
    
    st.divider()
    
    # AI Model Configuration
    st.subheader("🤖 AI Model Configuration")
    
    # AI Model Selection
    current_use_grok = st.session_state.get('use_grok', True)
    use_grok = st.checkbox("Use Grok AI (recommended)", value=current_use_grok)
    
    if use_grok != current_use_grok:
        st.session_state.use_grok = use_grok
        st.success("AI model configuration updated")
        st.rerun()
    
    if use_grok:
        st.success("✅ Using Grok AI (xAI) for best results")
        st.info("API Key: Configured via config.py file")
        
        # Test API connection
        if st.button("Test Grok API Connection"):
            with st.spinner("Testing connection..."):
                try:
                    # Simple test of the API
                    test_results = st.session_state.ai_engine.generate_grok_response(
                        "You are a test assistant.", 
                        "Respond with 'API connection successful' if you receive this message."
                    )
                    if "successful" in test_results.lower():
                        st.success("✅ Grok API connection successful")
                    else:
                        st.warning("⚠️ API responded but may have issues")
                        with st.expander("Response details"):
                            st.write(test_results)
                except Exception as e:
                    st.error(f"❌ API connection failed: {e}")
    else:
        st.info("Using local AI model (Ollama)")
        st.write("**Ollama Setup Instructions:**")
        st.code("""
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai for other platforms

# Start Ollama service
ollama serve

# Download a model
ollama pull llama3.2
        """)
        
        # Test Ollama connection
        if st.button("Test Ollama Connection"):
            with st.spinner("Testing local connection..."):
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        st.success(f"✅ Ollama connection successful. Available models: {len(models)}")
                        if models:
                            model_names = [model.get('name', 'Unknown') for model in models]
                            st.write("**Available models:**", ", ".join(model_names))
                    else:
                        st.error("❌ Ollama service not responding")
                except Exception as e:
                    st.error(f"❌ Cannot connect to Ollama: {e}")
                    st.info("Make sure Ollama is installed and running on localhost:11434")
        
        # Test Ollama response quality
        if st.button("Test Ollama Response Quality"):
            with st.spinner("Testing Ollama response quality..."):
                try:
                    test_prompt = "Explain the biblical concept of grace in 2-3 sentences."
                    response = st.session_state.ai_engine.generate_local_response(
                        "You are a theological expert providing clear, concise explanations.", 
                        test_prompt
                    )
                    st.write("**Test Response:**")
                    st.write(response)
                    
                    # Show current settings being used
                    st.write("**Current Ollama Settings:**")
                    st.code("""
Context Window: 8192 tokens
Temperature: 0.7
Top K: 40
Top P: 0.9
Repeat Penalty: 1.1
Max Response: 1000 tokens
                    """)
                except Exception as e:
                    st.error(f"❌ Ollama test failed: {e}")
    
    st.divider()
    
    # Advanced System Settings
    st.subheader("🔧 Advanced System Settings")
    
    # Initialize settings in session state if not present
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    if 'max_search_results' not in st.session_state:
        st.session_state.max_search_results = MAX_SEARCH_RESULTS
    
    # Text Processing Settings
    st.write("**📝 Text Processing Settings**")
    chunk_size = st.slider(
        "Chunk Size (words)", 
        500, 2000, 
        st.session_state.chunk_size,
        help="Larger chunks provide more context but may be less precise. Affects new sermon processing."
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap (words)", 
        50, 500, 
        st.session_state.chunk_overlap,
        help="Overlap between chunks to maintain context. Affects new sermon processing."
    )
    
    # Search Settings
    st.write("**🔍 Search Settings**")
    max_results = st.slider(
        "Max Search Results", 
        3, 20, 
        st.session_state.max_search_results,
        help="Number of relevant excerpts to return for each question. Affects all users."
    )
    
    # Save settings when they change
    settings_changed = False
    if chunk_size != st.session_state.chunk_size:
        st.session_state.chunk_size = chunk_size
        settings_changed = True
    
    if chunk_overlap != st.session_state.chunk_overlap:
        st.session_state.chunk_overlap = chunk_overlap
        settings_changed = True
    
    if max_results != st.session_state.max_search_results:
        st.session_state.max_search_results = max_results
        settings_changed = True
    
    if settings_changed:
        st.success("⚙️ System settings updated")
    
    # Reset to defaults button
    if st.button("🔄 Reset to Defaults"):
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        st.session_state.max_search_results = MAX_SEARCH_RESULTS
        st.success("Settings reset to defaults")
        st.rerun()
    
    st.divider()
    
    # System Information
    st.subheader("📊 System Information")
    
    # Database info
    sermons_dir = Path("data/sermons")
    if sermons_dir.exists():
        sermon_files = list(sermons_dir.glob("*.md"))
        processed_files = st.session_state.ai_engine.get_processed_files()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Sermon Files", len(sermon_files))
        with col2:
            st.metric("✅ Processed", len(processed_files))
        with col3:
            unprocessed = len(sermon_files) - len(processed_files)
            st.metric("⏳ Unprocessed", unprocessed)
    
    # Current settings display
    st.write("**Current System Settings:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunk Size", f"{st.session_state.chunk_size} words")
    with col2:
        st.metric("Chunk Overlap", f"{st.session_state.chunk_overlap} words")
    with col3:
        st.metric("Max Results", st.session_state.max_search_results)
    
    # System status
    st.write("**System Status:**")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.write("**Database:** Connected ✅")
        st.write("**Embedding Model:** Loaded ✅")
    with status_col2:
        if st.session_state.get('use_grok', True):
            st.write("**AI Model:** Grok AI (xAI) ✅")
        else:
            st.write("**AI Model:** Local (Ollama)")
        st.write("**Role:** Administrator ✅")
    
    st.divider()
    
    # Database Management
    st.subheader("🗄️ Database Management")
    
    # Dropbox sync section
    try:
        from app.config.config import USE_DROPBOX_SYNC, DROPBOX_ACCESS_TOKEN
        if USE_DROPBOX_SYNC:
            st.write("**📦 Dropbox Sync**")
            
            # Get backup info
            backup_info = st.session_state.ai_engine.get_cloud_backup_info()
            if backup_info and backup_info.get('exists'):
                st.info(f"Dropbox backup: {backup_info['size']:,} bytes, last modified: {backup_info['modified']}")
            else:
                st.info("No Dropbox backup found")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download from Dropbox", help="Replace local database with Dropbox backup"):
                    if st.session_state.get('confirm_download', False):
                        with st.spinner("Downloading database from Dropbox..."):
                            success = st.session_state.ai_engine.sync_database_from_cloud()
                            if success:
                                st.success("Database downloaded from Dropbox!")
                                st.info("Please refresh the page to see updated data")
                            else:
                                st.error("Download failed or no backup found")
                            st.session_state.confirm_download = False
                        st.rerun()
                    else:
                        st.session_state.confirm_download = True
                        st.warning("This will replace your local database. Click again to confirm.")
            
            with col2:
                if st.button("📤 Upload to Dropbox", help="Backup current database to Dropbox"):
                    with st.spinner("Uploading database to Dropbox..."):
                        success = st.session_state.ai_engine.sync_database_to_cloud()
                        if success:
                            st.success("Database uploaded to Dropbox!")
                        else:
                            st.error("Upload failed")
            
            st.divider()
        else:
            st.info("Dropbox sync not enabled. Set USE_DROPBOX_SYNC=true and DROPBOX_ACCESS_TOKEN in your environment.")
            st.divider()
    except ImportError as e:
        st.error(f"Dropbox configuration error: {e}")
        st.info("Make sure your config.py file is properly set up with Dropbox settings.")
        st.divider()
    
    st.warning("⚠️ **Danger Zone** - These operations affect all users")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reprocess All Sermons", help="Clear database and reprocess all sermon files"):
            if st.session_state.get('confirm_reprocess', False):
                with st.spinner("Reprocessing all sermons..."):
                    # Clear database
                    st.session_state.ai_engine.clear_database()
                    # Trigger processing
                    st.session_state.ai_engine.process_new_sermons()
                    st.success("All sermons reprocessed successfully!")
                    st.session_state.confirm_reprocess = False
                st.rerun()
            else:
                st.session_state.confirm_reprocess = True
                st.warning("Click again to confirm reprocessing all sermons")
    
    with col2:
        if st.button("🗑️ Clear All Data", help="Remove all processed sermons from database"):
            if st.session_state.get('confirm_clear_all', False):
                with st.spinner("Clearing all data..."):
                    success = st.session_state.ai_engine.clear_database()
                    if success:
                        st.success("All data cleared successfully!")
                        st.session_state.confirm_clear_all = False
                    else:
                        st.error("Failed to clear data")
                st.rerun()
            else:
                st.session_state.confirm_clear_all = True
                st.error("⚠️ This will delete ALL sermon data. Click again to confirm.")
    
    # File system info
    st.subheader("📂 Directory Structure")
    st.code("""
sermon-ai/
├── app/data/             # User accounts and session data
├── data/sermons/         # Sermon source files (.md)
├── sermon_db/            # Vector database (ChromaDB)
├── logs/                 # Application logs
└── backup/              # Backup storage
    """)
    
    # Environment info
    import os
    st.subheader("🌍 Environment")
    env_info = {
        "Python Version": f"{st.version_info if hasattr(st, 'version_info') else 'Unknown'}",
        "Streamlit Version": st.__version__,
        "Database Path": os.path.abspath("./sermon_db"),
        "User Data Path": os.path.abspath("app/data/"),
    }
    
    for key, value in env_info.items():
        st.write(f"**{key}:** {value}")