# UWCI Sermon AI

A smart AI system to ingest sermon transcripts and answer detailed questions about their content with enhanced search capabilities.

## Features

- **Enhanced Search**: Hybrid semantic + keyword search with relevance ranking and context expansion
- **Secure Authentication**: Production-ready user management with role-based access control
- **AI-Powered Chat**: Conversational interface with your sermon library using Grok AI or local Ollama
- **Smart Processing**: Automatic sermon metadata extraction and intelligent chunking
- **Pastor Filtering**: Focus searches on specific pastors' teachings
- **Comprehensive Logging**: Detailed activity tracking and error monitoring

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create directories:**
   ```bash
   mkdir -p app/data data/sermons logs app/utils
   ```

3. **Configure API keys:**
   ```bash
   cp app/config/config.template.py app/config/config.py
   # Edit app/config/config.py with your Grok API key
   ```

4. **Add sermon files:**
   Place sermon transcript files (`.md` format) in `data/sermons/`

5. **Start application:**
   ```bash
   streamlit run main.py
   ```

## First-Time Login

1. Check `app/data/initial_admin_password.txt` for admin credentials
2. Login and immediately change the admin password
3. Create additional user accounts via Settings > User Management
4. Delete the initial password file

## Enhanced Search Features

The application includes advanced search capabilities:

- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Relevance Re-ranking**: Analyzes query intent and content relevance
- **Context Expansion**: Includes surrounding sermon content for better understanding
- **Smart Filtering**: Weights results based on topics, pastors, and user preferences
- **Search Explanations**: Shows how results were determined

Configure search behavior in the Settings tab or through the sidebar during chat.

## Project Structure

```
sermon-ai/
├── app/
│   ├── auth.py               # Authentication system
│   ├── ai_engine.py          # Enhanced AI processing engine
│   ├── sermon_processor.py   # Sermon processing and metadata extraction
│   ├── utils/                # Enhanced utilities
│   │   ├── error_handler.py  # Error handling and logging
│   │   ├── logger.py         # Comprehensive logging system
│   │   ├── relevance_reranker.py  # Search relevance scoring
│   │   ├── context_expander.py    # Context expansion for results
│   │   └── metadata_filter.py     # Advanced filtering and weighting
│   ├── ui/                   # User interface components
│   └── config/
│       ├── config.template.py # Template (safe for GitHub)
│       └── config.py         # Your actual config (excluded from git)
├── data/
│   └── sermons/              # Sermon files (.md format)
├── logs/                     # Application logs
├── sermon_db/                # ChromaDB vector database
└── main.py                   # Main application entry point
```

## Sermon File Format

Sermon files should be in Markdown format:

```markdown
# Sermon Title
Pastor Name
Date or "Watch"
Additional date if "Watch" was used

Sermon content goes here...
```

## Environment Variables

Set these in your `.env` file or system environment:

```bash
GROK_API_KEY=your-actual-grok-api-key
ENABLE_ENHANCED_LOGGING=true
ENABLE_RATE_LIMITING=true
DATABASE_PATH=./sermon_db
USER_DATA_PATH=app/data/users.json
SESSION_TIMEOUT=86400
```

## Production Deployment

### Security Hardening
```bash
# Set proper file permissions
chmod 700 app/data/
chmod 600 app/data/users.json
chmod 600 app/config/config.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Critical Data Backup
- `app/data/users.json` - User accounts
- `data/sermons/` - Sermon files  
- `sermon_db/` - Vector database
- `app/config/config.py` - Configuration

## Monitoring and Logs

The application creates structured log files in the `logs/` directory:
- `app_YYYYMMDD.log` - General application events
- `auth_YYYYMMDD.log` - Authentication and security events
- `database_YYYYMMDD.log` - Database operations
- `ai_YYYYMMDD.log` - AI model operations
- `activity_YYYYMMDD.log` - User activities
- `errors_YYYYMMDD.log` - Error details

Monitor system health through:
- Log file analysis
- Admin panel session statistics
- Search performance metrics

## Troubleshooting

### Common Issues

**"Initial admin password file not found"**
- Delete `app/data/users.json` to regenerate initial admin

**"Account locked" errors**  
- Admin can reset failed attempts in user management
- Wait 15 minutes for automatic unlock

**Enhanced search not working**
- Ensure all utility files are in `app/utils/` 
- Check logs for import errors
- Basic search will work as fallback

### Support Commands
```bash
# Reset all users (backup first!)
rm app/data/users.json

# Check user file
cat app/data/users.json | python -m json.tool

# View recent logs  
tail -f logs/sermon_ai_$(date +%Y%m%d).log
```

## Security Features

- PBKDF2 password hashing with 100,000 iterations
- Rate limiting with account lockout protection
- Role-based access control (user/administrator)
- Session management with automatic cleanup
- User activity tracking and audit logs
- Comprehensive error handling and logging

## License

This project is for internal use by UWCI.