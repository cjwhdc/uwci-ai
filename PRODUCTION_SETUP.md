# Production Setup Guide

This guide will help you deploy the UWCI Sermon AI application securely in a production environment.

## Security Features Added

### 🔐 Enhanced Authentication
- **Secure password hashing** using PBKDF2 with 100,000 iterations
- **Salt-based hashing** to prevent rainbow table attacks
- **Rate limiting** with account lockout after 5 failed attempts
- **Session management** with proper cleanup
- **Role-based access control** (user/administrator)

### 🛡️ User Management
- **Admin interface** for creating and managing users
- **Password complexity requirements** (minimum 8 characters)
- **User activity tracking** (last login, failed attempts)
- **Secure user data storage** in encrypted JSON format

## Pre-Production Checklist

### 1. Environment Setup
```bash
# Create production directory structure
mkdir -p app/data
mkdir -p data/sermons
mkdir -p logs

# Set proper permissions
chmod 700 app/data
chmod 755 data/sermons
```

### 2. Security Configuration

#### Update API Keys
Edit `app/config/config.py`:
```python
# Use environment variables in production
import os

GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-production-api-key')
```

#### Set Environment Variables
```bash
# Add to your production environment
export GROK_API_KEY="your-actual-grok-api-key"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 3. Initial Admin Setup

1. **Run the application for the first time:**
   ```bash
   streamlit run main.py
   ```

2. **Check for initial admin credentials:**
   ```bash
   cat app/data/initial_admin_password.txt
   ```

3. **Login with admin credentials and immediately:**
   - Change the admin password
   - Create additional user accounts
   - Delete the initial password file

### 4. User Account Setup

#### Admin Tasks:
1. Login as admin
2. Go to Settings → User Management
3. Create user accounts for each person who needs access
4. Assign appropriate roles (user/administrator)

#### Security Best Practices:
- Use strong, unique passwords (minimum 8 characters)
- Regularly review user accounts and remove unused ones
- Monitor failed login attempts
- Change passwords if compromise is suspected

## Production Deployment Options

### Option 1: Local Server Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run with production settings
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t sermon-ai .
docker run -p 8501:8501 -v $(pwd)/data:/app/data sermon-ai
```

### Option 3: Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy with secrets management

## Security Hardening

### 1. File Permissions
```bash
# Set restrictive permissions on sensitive files
chmod 600 app/data/users.json
chmod 600 app/config/config.py
chmod 700 app/data/
```

### 2. Network Security
- Use HTTPS in production (configure reverse proxy)
- Restrict access to specific IP ranges if needed
- Consider VPN access for sensitive deployments

### 3. Regular Maintenance
- **Backup user data** regularly
- **Monitor login attempts** and failed authentications
- **Update dependencies** regularly for security patches
- **Review user accounts** quarterly

## Monitoring and Logging

### 1. Add Logging Configuration
Create `app/utils/logging.py`:
```python
import logging
import os
from datetime import datetime

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/sermon_ai_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
```

### 2. Monitor Key Events
- User login/logout events
- Failed authentication attempts
- File processing activities
- System errors and exceptions

## Backup Strategy

### 1. Critical Data to Backup
- `app/data/users.json` - User accounts
- `data/sermons/` - Sermon files
- `sermon_db/` - Vector database
- `app/config/config.py` - Configuration

### 2. Backup Script Example
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/sermon_ai_$DATE"

mkdir -p $BACKUP_DIR
cp -r app/data $BACKUP_DIR/
cp -r data/sermons $BACKUP_DIR/
cp -r sermon_db $BACKUP_DIR/
tar -czf "$BACKUP_DIR.tar.gz" $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

## Troubleshooting

### Common Issues

1. **"Initial admin password file not found"**
   - Delete `app/data/users.json` to regenerate initial admin
   - Check file permissions on `app/data/` directory

2. **"Account locked" errors**
   - Admin can reset failed attempts in user management
   - Wait 15 minutes for automatic unlock

3. **Database connection errors**
   - Check ChromaDB permissions
   - Ensure sufficient disk space

### Support Commands
```bash
# Reset all users (DESTRUCTIVE - backup first!)
rm app/data/users.json

# Check user file
cat app/data/users.json | python -m json.tool

# View recent logs
tail -f logs/sermon_ai_$(date +%Y%m%d).log
```

## Updates and Maintenance

### Regular Updates
1. Backup current installation
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Test in staging environment
4. Deploy to production
5. Verify all functionality

### Security Updates
- Monitor for security advisories for dependencies
- Update Streamlit and other packages regularly
- Review and update authentication settings annually

---

## Quick Start Checklist

- [ ] Install dependencies
- [ ] Set environment variables
- [ ] Run application first time
- [ ] Login with initial admin credentials
- [ ] Change admin password
- [ ] Create user accounts
- [ ] Delete initial password file
- [ ] Test login with regular user account
- [ ] Set up backups
- [ ] Configure monitoring
- [ ] Deploy to production environment

Your UWCI Sermon AI application is now production-ready with enterprise-grade security!