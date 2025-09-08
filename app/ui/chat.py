import streamlit as st
import re
from datetime import datetime

def show_chat_tab():
    """Display the chat interface tab"""
    
    # Set flag to show conversation settings in sidebar
    st.session_state.show_chat_filters = True
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
        # Get user's first name for personalized greeting
        try:
            from app.auth import user_manager
            profile = user_manager.get_user_profile(st.session_state.username)
            first_name = profile.get('first_name', '').strip()
            
            if first_name:
                greeting_name = first_name
            else:
                greeting_name = st.session_state.username
        except:
            # Fallback if profile lookup fails
            greeting_name = st.session_state.username
        
        # Add welcome message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Hello {greeting_name}! I'm here to help you explore your sermon library. I can answer questions about what different pastors have taught, find related Bible passages, and help you discover insights across your collection. What would you like to discuss?",
            "timestamp": datetime.now()
        })
    
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
                                st.write(f"**From {excerpt['pastor']} in \"{excerpt['title']}\":**")
                                st.write(excerpt['content'])
                                st.divider()
    
    # Chat input - add unique key based on tab context to prevent duplicates
    user_input = st.chat_input("Ask about your sermons...", key="ask_questions_chat_input")
    
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
            
            # Get pastor filter from sidebar if it exists
            pastor_filter = None
            if 'global_pastor_filter' in st.session_state:
                filter_value = st.session_state.global_pastor_filter
                if filter_value != "All Pastors":
                    pastor_filter = filter_value
            
            # Search for relevant content
            results = st.session_state.ai_engine.search_sermons(user_input, pastor_filter)
            
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
                            "title": result['metadata'].get('title', result['metadata']['filename']),
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