import streamlit as st
from pathlib import Path
from app.auth import change_password_form, is_admin

def show_settings_tab():
    """Display the user settings tab"""
    
    # Clear chat filters flag when not in chat tab
    if 'show_chat_filters' in st.session_state:
        st.session_state.show_chat_filters = False
    
    st.header("Settings")
    
    # User account settings
    st.subheader("Account Settings")
    
    # Password change form for all users
    change_password_form()
    
    st.subheader("User Preferences")
    st.write("Personal settings and preferences will be available here in future updates.")
    
    # System status with connection tests
    st.subheader("System Status")
    
    # Test Grok connection
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.get('grok_api_key'):
            st.write("**Grok AI:** Configured")
        else:
            st.write("**Grok AI:** Not configured")
    with col2:
        if st.button("Test", key="test_grok", help="Test Grok API connection"):
            with st.spinner("Testing Grok..."):
                try:
                    test_result = st.session_state.ai_engine.generate_grok_response(
                        "You are a test assistant.", 
                        "Respond with 'Connection successful' if you receive this message."
                    )
                    if "successful" in test_result.lower():
                        st.success("Grok connection successful")
                    else:
                        st.warning("Grok responded but may have issues")
                except Exception as e:
                    st.error(f"Grok connection failed: {str(e)[:50]}...")
    
    # Test Ollama connection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Ollama (Local AI):** Available")
    with col2:
        if st.button("Test", key="test_ollama", help="Test Ollama connection"):
            with st.spinner("Testing Ollama..."):
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        st.success(f"Ollama connected ({len(models)} models)")
                    else:
                        st.error("Ollama service not responding")
                except Exception as e:
                    st.error(f"Ollama connection failed: {str(e)[:50]}...")
    
    # Test Dropbox connection
    col1, col2 = st.columns([3, 1])
    with col1:
        try:
            from app.config.config import USE_DROPBOX_SYNC, DROPBOX_ACCESS_TOKEN
            if USE_DROPBOX_SYNC and DROPBOX_ACCESS_TOKEN:
                st.write("**Dropbox Sync:** Enabled")
            else:
                st.write("**Dropbox Sync:** Disabled")
        except ImportError:
            st.write("**Dropbox Sync:** Not configured")
    
    with col2:
        if st.button("Test", key="test_dropbox", help="Test Dropbox connection"):
            with st.spinner("Testing Dropbox..."):
                try:
                    from app.config.config import USE_DROPBOX_SYNC, DROPBOX_ACCESS_TOKEN
                    if USE_DROPBOX_SYNC and DROPBOX_ACCESS_TOKEN:
                        import dropbox
                        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
                        account = dbx.users_get_current_account()
                        st.success(f"Dropbox connected: {account.name.display_name}")
                    else:
                        st.info("Dropbox sync not enabled")
                except ImportError:
                    st.error("Dropbox not configured")
                except Exception as e:
                    st.error(f"Dropbox connection failed: {str(e)[:50]}...")
    
    # Database status (always show)
    st.write("**Database:** Connected")
    
    st.divider()