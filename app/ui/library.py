import streamlit as st
import pandas as pd
from pathlib import Path
from app.sermon_processor import SermonProcessor
from app.auth import is_admin
from collections import defaultdict

def show_library_tab():
    """Display the library management tab"""
    
    # Clear chat filters flag when not in chat tab
    if 'show_chat_filters' in st.session_state:
        st.session_state.show_chat_filters = False
    
    # Sermons Database section - visible to all users
    st.subheader("Sermons Database")
    
    try:
        # Get all data from the database
        all_results = st.session_state.ai_engine.collection.get()
        
        if all_results and all_results['metadatas']:
            # Group by filename to get unique sermons
            sermons_by_file = defaultdict(list)
            for metadata in all_results['metadatas']:
                filename = metadata.get('filename', 'Unknown')
                sermons_by_file[filename].append(metadata)
            
            # Create summary data for each sermon
            sermon_data = []
            total_chunks = 0
            
            for filename, chunks in sermons_by_file.items():
                # Use the first chunk's metadata for sermon-level info
                first_chunk = chunks[0]
                pastor = first_chunk.get('pastor', 'Unknown')
                title = first_chunk.get('title', 'Unknown')
                date = first_chunk.get('date', 'Unknown')
                word_count = first_chunk.get('word_count', 0)
                chunk_count = len(chunks)
                total_chunks += chunk_count
                
                # If title is still Unknown, use filename without .md extension
                if title == 'Unknown' or not title:
                    base_filename = filename.replace('.md', '').replace('.txt', '')
                    # Clean up filename to make it more readable as title
                    clean_title = base_filename.replace('_', ' ').replace('-', ' ')
                    # Capitalize first letter of each word
                    title = ' '.join(word.capitalize() for word in clean_title.split())
                
                sermon_data.append({
                    'Title': title,
                    'Pastor': pastor,
                    'Date': date,
                    'Words': word_count,
                    'Chunks': chunk_count,
                    'Filename': filename
                })
            
            # Sort by date (newest first)
            try:
                from app.sermon_processor import SermonProcessor
                processor = SermonProcessor()
                sermon_data.sort(key=lambda x: processor.parse_date_for_sorting(x['Date']), reverse=True)
            except Exception:
                # Fallback to title sorting if date sorting fails
                sermon_data.sort(key=lambda x: x['Title'])
            
            # Display the sermons table
            if sermon_data:
                df_sermons = pd.DataFrame(sermon_data)
                # Reorder columns for better display (removed Filename column)
                display_columns = ['Title', 'Pastor', 'Date', 'Words', 'Chunks']
                df_sermons = df_sermons[display_columns]
                st.dataframe(df_sermons, width='stretch')
            
            # Admin action buttons - moved below the database display
            if is_admin():
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Process New Sermons", type="primary", help="Scan for and process any new sermon files"):
                        with st.spinner("Scanning and processing sermon files..."):
                            st.session_state.ai_engine.process_new_sermons()
                
                with col2:
                    if st.button("Clear Database", type="secondary", help="Remove all processed sermons from database"):
                        # Add confirmation
                        if st.session_state.get('confirm_clear', False):
                            with st.spinner("Clearing database..."):
                                result = st.session_state.ai_engine.clear_database()
                                if isinstance(result, tuple):
                                    success, message = result
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                                    st.session_state.confirm_clear = False
                                else:
                                    # Handle old return format for compatibility
                                    if result:
                                        st.success("Database cleared successfully!")
                                    else:
                                        st.error("Failed to clear database")
                                    st.session_state.confirm_clear = False
                        else:
                            st.session_state.confirm_clear = True
                            st.warning("Click again to confirm database clearing")
        else:
            st.info("No sermons found in the database.")
            
            # For admins, show instruction to process sermons and buttons
            if is_admin():
                st.write("To add sermons to the database:")
                st.write("1. Place sermon files (.md format) in the data/sermons folder")
                st.write("2. Click 'Process New Sermons' button below")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Process New Sermons", type="primary", help="Scan for and process any new sermon files"):
                        with st.spinner("Scanning and processing sermon files..."):
                            st.session_state.ai_engine.process_new_sermons()
                
                with col2:
                    if st.button("Clear Database", type="secondary", help="Remove all processed sermons from database"):
                        # Add confirmation
                        if st.session_state.get('confirm_clear', False):
                            with st.spinner("Clearing database..."):
                                result = st.session_state.ai_engine.clear_database()
                                if isinstance(result, tuple):
                                    success, message = result
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                                    st.session_state.confirm_clear = False
                                else:
                                    # Handle old return format for compatibility
                                    if result:
                                        st.success("Database cleared successfully!")
                                    else:
                                        st.error("Failed to clear database")
                                    st.session_state.confirm_clear = False
                        else:
                            st.session_state.confirm_clear = True
                            st.warning("Click again to confirm database clearing")
    
    except Exception as e:
        error_msg = str(e)
        if "Missing metadata segment" in error_msg or "Missing field" in error_msg:
            st.error("Database corruption detected: Missing metadata segments")
            st.warning("The database appears to be corrupted. This can happen after importing from a backup or if the database was interrupted during writing.")
            
            if is_admin():
                st.write("**Recovery Options:**")
                st.write("1. **Clear and rebuild database** - This will remove all current data and reprocess sermons from files")
                st.write("2. **Restart the application** - Sometimes helps with temporary corruption")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear and Rebuild Database", help="Clear corrupted database and reprocess all sermon files"):
                        if st.session_state.get('confirm_rebuild', False):
                            with st.spinner("Rebuilding database..."):
                                # Clear the corrupted database
                                result = st.session_state.ai_engine.clear_database()
                                if isinstance(result, tuple):
                                    success, message = result
                                else:
                                    success = result
                                
                                if success:
                                    # Process sermons again
                                    st.session_state.ai_engine.process_new_sermons()
                                    st.success("Database rebuilt successfully!")
                                else:
                                    st.error("Failed to clear corrupted database. Try restarting the application.")
                                st.session_state.confirm_rebuild = False
                        else:
                            st.session_state.confirm_rebuild = True
                            st.warning("This will delete all current data and reprocess from files. Click again to confirm.")
                
                with col2:
                    st.info("If the problem persists, check that all sermon files are properly formatted and not corrupted.")
            else:
                st.info("Contact your administrator to fix the database corruption.")
        else:
            st.error(f"Error accessing database: {error_msg}")
            st.info("Make sure the AI engine is properly initialized.")
    
    st.divider()
    
    # Instructions - different based on user role
    if is_admin():
        st.subheader("How Manual Processing Works (Admin)")
        st.markdown("""
        **Database-Driven Display:**
        - The table above shows sermons that are actually processed and searchable
        - Data comes directly from the ChromaDB vector database
        - Each sermon is split into "chunks" for better AI processing
        
        **Processing New Sermons:**
        1. **Place sermon files** in the `data/sermons/` folder (only .md files are processed)
        2. **Click "Process New Sermons"** to scan for and process any new files
        3. **Files are chunked and indexed** for AI analysis and search
        4. **Already processed files are skipped** to avoid duplicates
        5. **Use "Clear Database"** if you need to start fresh (requires confirmation)
        
        **Benefits of manual processing:**
        - Control over when processing happens
        - No unexpected processing during startup
        - Better visibility into what's being processed
        - Faster application startup time
        """)
    else:
        st.subheader("Sermon Library Information")
        st.markdown("""
        This shows sermons that have been processed and are available for search:
        
        - **Sermons in Database**: Total number of processed sermons you can search
        - **Total Chunks**: Number of searchable text segments (sermons are split for better AI analysis)
        - **Pastors Featured**: Number of different pastors in the searchable library
        
        The data shown comes directly from the processed database, so it reflects exactly what's available for your questions and searches.
        
        Contact your administrator if you need new sermons added or processed.
        """)