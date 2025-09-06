import streamlit as st
import re
from datetime import datetime

def show_chat_tab():
    """Display the chat interface tab"""
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Hello {st.session_state.username}! I'm here to help you explore your sermon library. I can answer questions about what different pastors have taught, find related Bible passages, and help you discover insights across your collection. What would you like to discuss?",
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
                                st.write(f"**From {excerpt['pastor']} in \"{excerpt['title']}\":**")
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