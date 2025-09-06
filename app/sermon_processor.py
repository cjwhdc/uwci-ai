import re
from typing import List, Dict, Any
from datetime import datetime

class SermonProcessor:
    """Handle sermon processing and metadata extraction"""
    
    def __init__(self):
        self.translations = {
            'AMP': 'Amplified Bible',
            'NLT': 'New Living Translation'
        }
        self.bible_data = {}
        self.book_names = {}
    
    def clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting from text"""
        if not text:
            return text
        
        # Remove leading # symbols and whitespace
        text = re.sub(r'^#+\s*', '', text)
        # Remove markdown formatting characters
        text = re.sub(r'[*_`~]', '', text)
        # Remove remaining special characters used in markdown
        text = text.strip('# *_-=+')
        return text.strip()
    
    def parse_date_for_sorting(self, date_str: str) -> datetime:
        """Parse date string and return datetime object for sorting"""
        if not date_str or date_str == 'Unknown':
            return datetime.min  # Put unknown dates at the end
        
        # Clean the date string
        clean_date = self.clean_markdown_formatting(date_str)
        
        # Try different date formats
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}-\d{1,2}-\d{4})',  # MM-DD-YYYY
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, clean_date, re.IGNORECASE)
            if match:
                date_text = match.group(1)
                try:
                    # Try different parsing formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%B %d, %Y', '%B %d %Y', '%d %b %Y']:
                        try:
                            return datetime.strptime(date_text, fmt)
                        except ValueError:
                            continue
                except ValueError:
                    pass
        
        # If no pattern matches, return minimum date
        return datetime.min
    
    def extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from sermon content"""
        metadata = {
            'filename': filename,
            'pastor': 'Unknown',
            'date': 'Unknown',
            'title': 'Unknown',
            'word_count': len(content.split())
        }
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Extract title (first line, cleaned up)
        if lines:
            title = lines[0].strip()
            title = self.clean_markdown_formatting(title)
            if title and len(title) > 0:
                metadata['title'] = title
        
        # If title is still Unknown, use filename without .md extension
        if metadata['title'] == 'Unknown' or not metadata['title']:
            base_filename = filename.replace('.md', '').replace('.txt', '')
            # Clean up filename to make it more readable as title
            clean_title = base_filename.replace('_', ' ').replace('-', ' ')
            # Capitalize first letter of each word
            clean_title = ' '.join(word.capitalize() for word in clean_title.split())
            metadata['title'] = clean_title
        
        # For your format: title, pastor, [optional "Watch"], date
        if len(lines) >= 3:
            # Skip the first line (title) and take the second line as pastor
            potential_pastor = lines[1]
            
            # Clean up the pastor name
            pastor_name = self.clean_markdown_formatting(potential_pastor)
            pastor_name = re.sub(r'^(pastor|rev|dr|mr|mrs)\.?\s*', '', pastor_name, flags=re.IGNORECASE)
            pastor_name = re.sub(r'\s*(sermon|message|teaching).*$', '', pastor_name, flags=re.IGNORECASE)
            
            # Validate it's a reasonable pastor name (not too long, not empty)
            if len(pastor_name) > 0 and len(pastor_name) < 50:
                metadata['pastor'] = pastor_name
            
            # Check if third line is "Watch" or similar, if so, date is on line 4
            if len(lines) >= 4 and lines[2].strip().lower() in ['watch', '"watch"', "'watch'"]:
                potential_date = lines[3]
            else:
                # Third line should be the date
                potential_date = lines[2]
            
            # Clean markdown characters from date and remove # symbols
            clean_date = self.clean_markdown_formatting(potential_date)
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
                        pastor_name = self.clean_markdown_formatting(pastor_name)
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
                clean_line = self.clean_markdown_formatting(line)
                for pattern in date_patterns:
                    match = re.search(pattern, clean_line, re.IGNORECASE)
                    if match:
                        metadata['date'] = match.group(1)
                        break
                if metadata['date'] != 'Unknown' and metadata['date']:
                    break
        
        return metadata
    
    def chunk_sermon(self, content: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split sermon into overlapping chunks"""
        # Import here to avoid circular imports
        import streamlit as st
        
        # Use session state settings if available, otherwise use defaults
        if chunk_size is None:
            chunk_size = st.session_state.get('chunk_size', 1000)
        if overlap is None:
            overlap = st.session_state.get('chunk_overlap', 200)
        
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