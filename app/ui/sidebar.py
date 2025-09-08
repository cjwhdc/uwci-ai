import streamlit as st
import re
from datetime import datetime

def show_sidebar():
    """Display the modern sidebar with all sections"""
    
    with st.sidebar:
        # User greeting section (no header)
        try:
            from app.auth import user_manager
            profile = user_manager.get_user_profile(st.session_state.username)
            first_name = profile.get('first_name', '').strip()
            
            if first_name:
                user_display = first_name
            else:
                user_display = st.session_state.username
        except:
            user_display = st.session_state.username
        
        # User info card
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: rgba(28, 131, 225, 0.1);
                border-left: 4px solid #1c83e1;
                padding: 12px;
                border-radius: 4px;
                margin: 16px 0;
            ">
                <div style="font-size: 14px; color: #666; margin-bottom: 4px;">Welcome back</div>
                <div style="font-size: 16px; font-weight: 600; color: #1c83e1;">{user_display}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Conversation Settings section
        _show_conversation_settings()
        
        st.markdown("---")
        
        # System status section
        _show_system_status()
        
        st.markdown("---")
        
        # Quick stats section
        _show_library_stats()
        
        # Navigation help
        _show_help_section()
        
        # Push logout button to very bottom
        st.markdown("<br>" * 10, unsafe_allow_html=True)
        
        # Logout button at very bottom
        st.markdown("---")
        if st.button("Sign Out", type="secondary", use_container_width=True):
            from main import logout_persistent
            logout_persistent()

def _show_conversation_settings():
    """Show conversation settings section"""
    st.markdown("**Conversation Settings**")
    
    # Get pastors for filter
    try:
        if (hasattr(st.session_state, 'ai_engine') and 
            st.session_state.ai_engine and 
            hasattr(st.session_state.ai_engine, 'collection') and 
            st.session_state.ai_engine.collection):
            
            all_results = st.session_state.ai_engine.collection.get()
            
            # Check if we actually have data
            if all_results and all_results.get('metadatas') and len(all_results['metadatas']) > 0:
                raw_pastors = [meta.get('pastor', 'Unknown') for meta in all_results['metadatas']]
                
                # Clean up pastor names
                cleaned_pastors = set()
                for pastor in raw_pastors:
                    clean_pastor = re.sub(r'^#+\s*', '', pastor)
                    clean_pastor = re.sub(r'[*_`~]', '', clean_pastor)
                    clean_pastor = clean_pastor.strip('# *_-=+')
                    if clean_pastor and clean_pastor != 'Unknown':
                        cleaned_pastors.add(clean_pastor)
                
                pastors = ["All Pastors"] + sorted(list(cleaned_pastors))
                pastor_filter = st.selectbox("Focus on specific pastor", pastors, key="global_pastor_filter")
                
                # Show clear conversation button only when in chat tab
                if 'show_chat_filters' in st.session_state and st.session_state.show_chat_filters:
                    if st.button("Clear Conversation", key="global_clear_chat"):
                        if 'chat_history' in st.session_state:
                            st.session_state.chat_history = []
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": "Conversation cleared. What would you like to discuss about your sermons?",
                                "timestamp": datetime.now()
                            })
                        st.rerun()
            else:
                st.markdown("<div style='font-size: 12px; color: #666;'>No sermon data found</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 12px; color: #666;'>AI engine initializing...</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div style='font-size: 12px; color: #666;'>Error loading pastors: {str(e)[:50]}...</div>", unsafe_allow_html=True)

def _show_system_status():
    """Show system status section with connection indicators"""
    st.markdown("**System Status**")
    
    # AI Model status with colored indicator
    if st.session_state.get('use_grok', False):
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; margin-right: 8px;"></div>
            <span style="font-size: 14px;">Grok AI Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 8px; height: 8px; background-color: #3b82f6; border-radius: 50%; margin-right: 8px;"></div>
            <span style="font-size: 14px;">Local AI (Ollama)</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Database status
    st.markdown("""
    <div style="display: flex; align-items: center; margin: 8px 0;">
        <div style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; margin-right: 8px;"></div>
        <span style="font-size: 14px;">Database Connected</span>
    </div>
    """, unsafe_allow_html=True)

def _show_library_stats():
    """Show library overview stats if available"""
    try:
        if hasattr(st.session_state, 'ai_engine') and st.session_state.ai_engine.collection:
            all_results = st.session_state.ai_engine.collection.get()
            if all_results and all_results['metadatas']:
                # Calculate quick stats
                total_chunks = len(all_results['metadatas'])
                unique_files = len(set(meta.get('filename', 'Unknown') for meta in all_results['metadatas']))
                
                st.markdown("**Library Overview**")
                
                # Stats grid
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Sermons",
                        value=str(unique_files),
                        help="Total sermons in database"
                    )
                with col2:
                    st.metric(
                        label="Sections",
                        value=f"{total_chunks:,}",
                        help="Searchable text segments"
                    )
                
                st.markdown("---")
    except:
        pass  # Don't show stats if there's an error

def _show_help_section():
    """Show navigation help section"""
    st.markdown("**Need Help?**")
    st.markdown("""
    <div style="font-size: 12px; color: #666; line-height: 1.4;">
    • <strong>Ask Questions:</strong> Search your sermon library<br>
    • <strong>Library:</strong> Manage sermon database<br>
    • <strong>Settings:</strong> Update your profile
    </div>
    """, unsafe_allow_html=True)