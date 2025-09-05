#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from sermon_qa_app import SermonProcessor, AIEngine

def bulk_upload_sermons(sermons_dir: Path):
    """Process all sermon files in directory"""
    processor = SermonProcessor()
    ai_engine = AIEngine()
    
    for sermon_file in sermons_dir.glob("*.md"):
        print(f"Processing {sermon_file.name}...")
        
        with open(sermon_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = processor.extract_metadata(content, sermon_file.name)
        chunks = processor.chunk_sermon(content)
        
        success = ai_engine.add_sermon(content, metadata, chunks)
        if success:
            print(f"✓ Successfully processed {sermon_file.name}")
        else:
            print(f"✗ Failed to process {sermon_file.name}")

if __name__ == "__main__":
    sermons_path = Path(__file__).parent.parent / "data" / "sermons"
    bulk_upload_sermons(sermons_path)