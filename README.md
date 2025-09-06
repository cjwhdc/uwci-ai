# UWCI Sermon AI

A smart AI system to ingest sermon transcripts and answer detailed questions about their content.

## Features

- **Secure Login**: Production-ready authentication with rate limiting and user management
- **Auto-Import**: Automatically process sermon files from the `data/sermons/` directory
- **AI-Powered Search**: Vector-based semantic search using ChromaDB
- **Conversational Interface**: Chat with your sermon library
- **Multiple AI Models**: Support for both Grok AI and local Ollama models
- **Pastor Filtering**: Focus on specific pastors' teachings
- **Sermon Titles**: References sermons by title instead of dates

## Security Notice

This application includes production-ready security features including user authentication, encrypted password storage, and rate limiting. Never commit sensitive configuration files to version control.

## Quick Setup (After GitHub Clone)

1. **Run the automated setup:**
   ```bash
   python setup.py
   ```

2. **Edit configuration:**
   - Add your Grok API key to `app/config/config.py` OR
   - Set environment variables in `.env` file

3. **Start the application:**
   ```bash
   streamlit run main.py
   ```

## Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create directories:**
   ```bash
   mkdir -p app/data data/sermons logs
   ```

3. **Configure API keys:**
   ```bash
   cp app/config/config.template.py app/config/config.py
   # Edit app/config/config.py with your actual Grok API key
   ```

4. **OR use environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

5. **Add sermon files:**
   Place your sermon transcript files (`.md` format) in `data/sermons/`

6. **Start application:**
   ```bash
   streamlit run main.py
   ```

## First-Time Login

1. Check `app/data/initial_admin_password.txt` for admin credentials
2. Login and immediately change the admin password
3. Create additional user accounts via Settings > User Management
4. Delete the initial password file

## Project Structure

```
sermon-ai/
├── app/
│   ├── __init__.py
│   ├── auth.py               # Production authentication system
│   ├── sermon_processor.py   # Sermon processing and metadata extraction
│   ├── ai_engine.py          # AI processing engine
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── login.py          # Login interface
│   │   ├── chat.py           # Chat interface
│   │   ├── library.py        # Library management
│   │   └── settings.py       # Settings and user management
│   └── config/
│       ├── __init__.py
│       ├── config.template.py # Template (safe for GitHub)
│       └── config.py         # Your actual config (excluded from git)
├── data/
│   └── sermons/              # Sermon files (excluded from git)
├── app/data/                 # User data storage (excluded from git)
├── sermon_db/                # ChromaDB storage (excluded from git)
├── main.py                   # Main application entry point
├── setup.py                  # Automated setup script
├── .env.example              # Environment variables template
├── .gitignore                # Protects sensitive files
├── requirements.txt
└── README.md
```

## Environment Variables

Set these in your `.env` file or system environment:

```bash
GROK_API_KEY=your-actual-grok-api-key
DATABASE_PATH=./sermon_db
USER_DATA_PATH=app/data/users.json
SESSION_TIMEOUT=3600
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

## Security Features

- **Secure authentication** with PBKDF2 password hashing
- **Rate limiting** with account lockout protection
- **Role-based access control** (user/administrator)
- **Session management** with proper cleanup
- **User activity tracking** and audit logs
- **Environment variable support** for secrets

## Production Deployment

See `PRODUCTION_SETUP.md` for detailed production deployment instructions including:
- Security hardening
- Backup strategies
- Monitoring setup
- Docker deployment
- Cloud deployment options

## Support

For production support and advanced configuration, refer to the production setup guide.

## License

This project is for internal use by UWCI.