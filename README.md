# Sermon AI Q&A System

A smart AI-powered system for analyzing sermon transcripts and answering detailed questions about their content. Built with Streamlit, ChromaDB, and OpenAI/Ollama integration.

## Features

- **Automatic sermon processing** from markdown files
- **AI-powered Q&A** with semantic search across all sermons
- **Pastor-specific queries** - ask what specific pastors said about topics
- **Bible integration** with XML Bible translations (AMP & NLT)
- **Intelligent chunking** for better context understanding
- **Dual AI support** - OpenAI or local Ollama models

## Quick Start

### 1. Installation

```bash
# Clone or download the project
git clone <your-repo-url>
cd sermon-ai

# Create directory structure
mkdir -p data/sermons data/bible

# Install dependencies
cd app
pip install -r requirements.txt
```

### 2. Add Your Content

Place your sermon files in the data directory:
```bash
# Add sermon transcripts (.md format)
cp your-sermons/*.md data/sermons/

# Add Bible translations (optional)
cp EnglishAmplifiedBible.xml data/bible/
cp EnglishNLTBible.xml data/bible/
```

### 3. Run the Application

```bash
cd app
streamlit run sermon_qa_app.py
```

The app will automatically:
- Process all sermon files on startup
- Create a vector database for semantic search
- Load Bible translations if present

## Usage

### Adding New Sermons
1. Drop `.md` files into `data/sermons/` folder
2. Go to "Sermon Library" tab
3. Click "Scan for New Sermons"

### Asking Questions
1. Go to "Ask Questions" tab
2. Optional: Filter by specific pastor
3. Ask detailed questions like:
   - "What did Pastor John say about faith?"
   - "What are the key themes about forgiveness?"
   - "Compare different pastors' views on deliverance"

### Bible Integration
1. Go to "Bible Integration" tab
2. Look up specific verses
3. Search for words/phrases across translations

## File Format Requirements

### Sermon Files (.md format)
```markdown
# Sermon Title
Pastor Name
Date (MM/DD/YYYY)

Sermon content goes here...
```

The system extracts:
- **Pastor name** from the second line
- **Date** from content (various formats supported)
- **Title** from filename or first line

### Bible Files (XML format)
Place XML Bible files in `data/bible/`:
- `EnglishAmplifiedBible.xml`
- `EnglishNLTBible.xml`

## AI Configuration

The system supports two AI options:

### OpenAI (Default)
- Hardcoded API key for immediate use
- Uses `gpt-4o-mini` model
- Best results for complex theological questions

### Local Ollama
1. Install Ollama: `brew install ollama`
2. Start service: `ollama serve`
3. Download model: `ollama pull llama3.2`
4. Switch to local AI in Settings tab

## Project Structure

```
sermon-ai/
├── app/
│   ├── sermon_qa_app.py      # Main application
│   └── requirements.txt      # Dependencies
├── data/
│   ├── sermons/             # Sermon .md files
│   └── bible/               # Bible XML files
├── sermon_db/               # Vector database (auto-created)
└── README.md
```

## Dependencies

- **Streamlit** - Web interface
- **ChromaDB** - Vector database for semantic search
- **OpenAI** - AI text generation
- **sentence-transformers** - Text embeddings
- **pandas** - Data management

## Advanced Features

### Semantic Search
The system uses advanced embeddings to find conceptually related content, not just keyword matches.

### Pastor Attribution
All responses include proper attribution to specific pastors with dates and sermon references.

### Intelligent Chunking
Long sermons are split into overlapping chunks to maintain context while enabling precise search.

### Bible Cross-Referencing
Automatically detects Bible references in sermons and links them to actual verses.

## Troubleshooting

### Common Issues

**"Sermon directory not found"**
- Create the `data/sermons` directory
- Ensure you're running from the correct location

**"No AI model available"**
- Check OpenAI API key is working
- For Ollama: run `ollama list` to see available models

**"Error processing sermons"**
- Check sermon file format (second line should be pastor name)
- Ensure files are UTF-8 encoded

### Database Management
- **Clear database**: Use "Clear Database" button in Sermon Library
- **Reset everything**: Delete `sermon_db/` folder and restart

## Configuration

### Settings Tab Options
- **AI Model Selection** - Switch between OpenAI and Ollama
- **Chunk Size** - Adjust for more/less context (500-2000 words)
- **Search Results** - Number of excerpts returned (3-20)

### File Paths
All paths are relative to the app directory:
- Sermons: `data/sermons/`
- Bible files: `data/bible/`
- Database: `sermon_db/`

## Security Note

The OpenAI API key is currently hardcoded in the source. For production use:
- Use environment variables
- Add source files to `.gitignore`
- Regenerate API keys if code is shared

## Support

For issues or questions:
1. Check the Settings tab for system status
2. Review file format requirements
3. Verify directory structure matches documentation

The system provides detailed error messages and status information to help diagnose issues.