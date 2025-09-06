import streamlit as st
import pandas as pd
from pathlib import Path
from app.sermon_processor import SermonProcessor
from app.auth import is_admin
from collections import defaultdict

def show_library_tab():
    """Display the library management tab"""
    st.header("Sermon Library Management")
    
    # Check if user is admin for processing functions
    if is_admin():
        # Action buttons at the top - Admin only
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process New Sermons", type="primary", help="Scan for and process any new sermon files"):
                with st.spinner("Scanning and processing sermon files..."):
                    st.session_state.ai_engine.process_new_sermons()
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
    else:
        # Non-admin users see a message
        st.info("Library Status - Contact administrator to process new sermons")
        st.divider()
    
    # Database content section - visible to all users
    st.subheader("Processed Sermons Database")
    
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
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sermons in Database", len(sermon_data))
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                unique_pastors = len(set(sermon['Pastor'] for sermon in sermon_data if sermon['Pastor'] != 'Unknown'))
                st.metric("Pastors Featured", unique_pastors)
            
            # Display the sermons table
            if sermon_data:
                df_sermons = pd.DataFrame(sermon_data)
                # Reorder columns for better display (removed Filename column)
                display_columns = ['Title', 'Pastor', 'Date', 'Words', 'Chunks']
                df_sermons = df_sermons[display_columns]
                st.dataframe(df_sermons, width='stretch')
            
            # Show file system comparison for admins
            if is_admin():
                st.divider()
                st.subheader("File System vs Database Comparison")
                
                sermons_dir = Path("data/sermons")
                if sermons_dir.exists():
                    filesystem_files = set(f.name for f in sermons_dir.glob("*.md"))
                    database_files = set(sermons_by_file.keys())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Files in filesystem but not in database:**")
                        missing_from_db = filesystem_files - database_files
                        if missing_from_db:
                            for file in sorted(missing_from_db):
                                st.write(f"- {file}")
                        else:
                            st.write("None")
                    
                    with col2:
                        st.write("**Files in database but not in filesystem:**")
                        missing_from_fs = database_files - filesystem_files
                        if missing_from_fs:
                            for file in sorted(missing_from_fs):
                                st.write(f"- {file}")
                        else:
                            st.write("None")
                else:
                    st.info("No data/sermons directory found on this system.")
        else:
            st.info("No sermons found in the database.")
            
            # For admins, show instruction to process sermons
            if is_admin():
                st.write("To add sermons to the database:")
                st.write("1. Place sermon files (.md format) in the data/sermons folder")
                st.write("2. Click 'Process New Sermons' button above")
    
    except Exception as e:
        st.error(f"Error accessing database: {e}")
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