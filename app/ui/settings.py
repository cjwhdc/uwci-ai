import streamlit as st
from pathlib import Path
from app.auth import change_password_form, is_admin

def show_settings_tab():
    """Display the user settings tab"""
    st.header("Settings")
    
    # User account settings
    st.subheader("Account Settings")
    
    # Password change form for all users
    change_password_form()
    
    st.divider()
    
    st.subheader("System Information")
    
    # User info
    st.info(f"Logged in as: {st.session_state.username} ({st.session_state.get('user_role', 'user')})")
    
    # Basic database info (read-only for regular users)
    sermons_dir = Path("data/sermons")
    processed_files = []  # Initialize with empty list
    
    if sermons_dir.exists():
        sermon_files = list(sermons_dir.glob("*.md"))
        processed_files = st.session_state.ai_engine.get_processed_files()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sermons", len(sermon_files))
        with col2:
            st.metric("Available for Search", len(processed_files))
        with col3:
            unprocessed = len(sermon_files) - len(processed_files)
            st.metric("Pending Processing", unprocessed)
        
        if unprocessed > 0:
            st.info(f"Contact an administrator to process {unprocessed} pending sermon(s).")
    
    # Current user settings (if any personal preferences are added later)
    st.subheader("User Preferences")
    st.write("Personal settings and preferences will be available here in future updates.")
    
    # System status (basic info)
    st.subheader("System Status")
    st.write("**Database:** Connected")
    st.write("**Search:** Available")
    
    if st.session_state.get('use_grok', True):
        st.write("**AI Model:** Grok AI")
    else:
        st.write("**AI Model:** Local (Ollama)")
    
    st.divider()
    
    st.subheader("Application Information")
    st.write("**UWCI Sermon AI** - Intelligent sermon analysis and search system")
    st.write("For technical support or feature requests, contact your administrator.")
    
    # Help section
    st.subheader("How to Use")
    st.markdown("""
    **Asking Questions:**
    - Use the "Ask Questions" tab to search your sermon library
    - Ask about specific topics, Bible passages, or pastor teachings
    - Use the pastor filter to focus on specific speakers
    
    **Tips for Better Results:**
    - Be specific in your questions
    - Try different phrasings if you don't find what you're looking for
    - Use theological terms and Bible references when relevant
    """)
    
    # Quick stats for user interest
    if len(processed_files) > 0:
        # Get some basic stats
        try:
            all_results = st.session_state.ai_engine.collection.get()
            if all_results['metadatas']:
                pastors = set(meta.get('pastor', 'Unknown') for meta in all_results['metadatas'])
                pastors.discard('Unknown')
                
                st.subheader("Library Overview")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Pastors Featured:** {len(pastors)}")
                with col2:
                    total_chunks = len(all_results['metadatas'])
                    st.write(f"**Searchable Sections:** {total_chunks:,}")
        except:
            pass  # Don't show stats if there's an error