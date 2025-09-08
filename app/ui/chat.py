import streamlit as st
from datetime import datetime

def show_chat_tab():
    """Display the chat interface tab with enhanced search"""
    
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
        welcome_message = f"Hello {greeting_name}! I'm here to help you explore your sermon library. I can answer questions about what different pastors have taught, find related Bible passages, and help you discover insights across your collection. What would you like to discuss?"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": welcome_message,
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
                                
                                # Show search quality info if available
                                if 'search_info' in excerpt:
                                    search_info = excerpt['search_info']
                                    score_parts = []
                                    
                                    if 'confidence' in search_info:
                                        confidence = search_info['confidence']
                                        if confidence > 0.8:
                                            score_parts.append("High confidence match")
                                        elif confidence > 0.6:
                                            score_parts.append("Good match")
                                        else:
                                            score_parts.append("Moderate match")
                                    
                                    if 'detected_topics' in search_info and search_info['detected_topics']:
                                        topics = search_info['detected_topics']
                                        score_parts.append(f"Topics: {', '.join(topics)}")
                                    
                                    if 'context_added' in search_info and search_info['context_added'] > 0:
                                        score_parts.append(f"Includes surrounding context")
                                    
                                    if score_parts:
                                        st.caption(" | ".join(score_parts))
                                
                                # Show content (use expanded if available, otherwise original)
                                content_to_show = excerpt.get('expanded_content', excerpt['content'])
                                if content_to_show != excerpt['content'] and st.session_state.get('show_expanded_content', True):
                                    st.write(content_to_show)
                                else:
                                    st.write(excerpt['content'])
                                
                                if i < len(message["excerpts"]) - 1:
                                    st.divider()
                    
                    # Show search explanation if enabled and available
                    if ("search_explanation" in message and 
                        st.session_state.get('show_search_explanation', False)):
                        with st.expander("How these results were found"):
                            st.info(message["search_explanation"])
    
    # Chat input
    user_input = st.chat_input("Ask about your sermons...", key="ask_questions_chat_input")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Process the question
        with st.spinner("Searching and analyzing..."):
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
            
            # Determine which search method to use
            search_method = "basic"
            results = []
            
            try:
                # Check if enhanced search is available and enabled
                search_options = getattr(st.session_state, 'search_options', {})
                use_enhanced = search_options.get('use_hybrid', True)
                
                if (hasattr(st.session_state.ai_engine, 'ultimate_search_sermons') and 
                    use_enhanced):
                    # Use enhanced search
                    results = st.session_state.ai_engine.ultimate_search_sermons(
                        user_input, 
                        pastor_filter,
                        search_options=search_options
                    )
                    search_method = "enhanced"
                elif hasattr(st.session_state.ai_engine, 'enhanced_search_sermons'):
                    # Use hybrid search if available
                    results = st.session_state.ai_engine.enhanced_search_sermons(
                        user_input, 
                        pastor_filter,
                        use_hybrid=True
                    )
                    search_method = "hybrid"
                else:
                    # Fall back to basic search
                    results = st.session_state.ai_engine.search_sermons(user_input, pastor_filter)
                    search_method = "basic"
                
            except Exception as e:
                # If enhanced search fails, fall back to basic
                try:
                    results = st.session_state.ai_engine.search_sermons(user_input, pastor_filter)
                    search_method = "basic_fallback"
                except Exception as e2:
                    st.error(f"Search error: {e2}")
                    results = []
                    search_method = "failed"
            
            if results:
                # Generate search explanation if enhanced search was used
                search_explanation = None
                if (search_method in ["enhanced", "hybrid"] and 
                    hasattr(st.session_state.ai_engine, 'get_search_explanation')):
                    try:
                        search_explanation = st.session_state.ai_engine.get_search_explanation(results, user_input)
                    except:
                        search_explanation = None
                
                # Generate conversational response
                use_grok = st.session_state.get('use_grok', True)
                response = st.session_state.ai_engine.generate_conversational_answer(
                    user_input, results, conversation_context, use_grok
                )
                
                # Prepare excerpts with enhanced information
                excerpts = []
                for result in results[:3]:  # Top 3 excerpts
                    excerpt = {
                        "pastor": result['metadata']['pastor'],
                        "title": result['metadata'].get('title', result['metadata']['filename']),
                        "content": result.get('original_content', result['content'])
                    }
                    
                    # Add expanded content if available
                    if 'expanded_content' in result:
                        excerpt['expanded_content'] = result['expanded_content']
                    
                    # Add search information if available
                    search_info = {}
                    for key in ['confidence', 'relevance_score', 'detected_topics', 'context_added']:
                        if key in result:
                            search_info[key] = result[key]
                    
                    if search_info:
                        excerpt['search_info'] = search_info
                    
                    excerpts.append(excerpt)
                
                # Add assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "excerpts": excerpts,
                    "timestamp": datetime.now(),
                    "search_method": search_method
                }
                
                if search_explanation:
                    assistant_message["search_explanation"] = search_explanation
                
                st.session_state.chat_history.append(assistant_message)
            else:
                # No relevant content found
                no_results_message = "I couldn't find relevant information in your sermon library to answer that question."
                
                # Add suggestions based on search method
                if search_method in ["enhanced", "hybrid"]:
                    no_results_message += " Try rephrasing your question, using different keywords, or asking about a broader topic."
                else:
                    no_results_message += " Could you try rephrasing it or asking about a different topic?"
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": no_results_message,
                    "timestamp": datetime.now(),
                    "search_method": search_method
                })
        
        st.rerun()