import hashlib
import streamlit as st
import os
import json
import secrets
from pathlib import Path
from typing import Dict, Optional
import time

class UserManager:
    """Production-ready user management system with enhanced user profiles"""
    
    def __init__(self, users_file: str = "app/data/users.json"):
        self.users_file = Path(users_file)
        self.backup_file = Path(users_file.replace('.json', '_backup.json'))
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_users()
        
    def _load_users(self):
        """Load users from secure file with backup recovery"""
        # Try main file first
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    content = f.read()
                    if content.strip():  # Check if file has content
                        self.users = json.loads(content)
                        # Create backup of working file
                        self._create_backup()
                        return
                    else:
                        print("Warning: users.json is empty")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading users.json: {e}")
        
        # Try backup file if main file failed
        if self.backup_file.exists():
            try:
                with open(self.backup_file, 'r') as f:
                    content = f.read()
                    if content.strip():
                        self.users = json.loads(content)
                        print("Recovered users from backup file")
                        # Restore main file from backup
                        self._save_users()
                        return
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading backup file: {e}")
        
        # If both files failed, create new user system
        print("Creating new user system")
        self.users = {}
        self._create_default_admin()
    
    def _create_backup(self):
        """Create backup copy of users file"""
        try:
            if self.users_file.exists():
                import shutil
                shutil.copy2(self.users_file, self.backup_file)
        except Exception as e:
            print(f"Failed to create backup: {e}")
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        if not self.users:
            # Check for environment variable first (for cloud deployment)
            initial_password = os.getenv('ADMIN_INITIAL_PASSWORD')
            
            if initial_password:
                # Use environment variable password (cloud deployment)
                self.add_user("admin", initial_password, "administrator")
                print("Admin account created using environment variable password.")
            else:
                # Generate a random password for local development
                temp_password = secrets.token_urlsafe(12)
                self.add_user("admin", temp_password, "administrator")
                
                # Try to save the temporary password to a file (local development)
                try:
                    temp_file = self.users_file.parent / "initial_admin_password.txt"
                    with open(temp_file, 'w') as f:
                        f.write(f"Initial admin password: {temp_password}\n")
                        f.write("Please login and change this password immediately!\n")
                        f.write("Delete this file after setting up your admin account.\n")
                    
                    print(f"Initial admin password saved to: {temp_file}")
                except Exception as e:
                    print(f"Could not save password file: {e}")
                    # If file creation fails (like on Streamlit Cloud), show in UI
                    st.error("**FIRST TIME SETUP REQUIRED**")
                    st.error(f"**Admin Password:** {temp_password}")
                    st.error("**Username:** admin")
                    st.error("Please save this password and login immediately to change it!")
                    st.error("This password will not be shown again.")
    
    def _save_users(self):
        """Save users to secure file with atomic write and backup"""
        try:
            # Write to temporary file first (atomic write)
            temp_file = self.users_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            
            # Move temp file to actual file (atomic operation)
            temp_file.replace(self.users_file)
            
            # Create backup
            self._create_backup()
            
        except Exception as e:
            print(f"Error saving users: {e}")
            # Try direct write as fallback
            try:
                with open(self.users_file, 'w') as f:
                    json.dump(self.users, f, indent=2)
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for better security
        import hashlib
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)  # 100,000 iterations
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        password_hash, _ = self.hash_password(password, salt)
        return password_hash == stored_hash
    
    def add_user(self, username: str, password: str, role: str = "user", first_name: str = "", last_name: str = "", email: str = "") -> bool:
        """Add new user with profile information"""
        username_lower = username.lower()
        if username_lower in self.users:
            return False
        
        password_hash, salt = self.hash_password(password)
        
        self.users[username_lower] = {
            "password_hash": password_hash,
            "salt": salt,
            "role": role,
            "created_at": time.time(),
            "last_login": None,
            "failed_attempts": 0,
            "locked_until": None,
            "display_name": username,  # Store original case for display
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }
        
        self._save_users()
        return True
    
    def update_user_profile(self, username: str, first_name: str = None, last_name: str = None, email: str = None) -> tuple:
        """Update user profile information"""
        username_lower = username.lower()
        if username_lower not in self.users:
            return False, "User not found"
        
        user = self.users[username_lower]
        
        if first_name is not None:
            user["first_name"] = first_name
        if last_name is not None:
            user["last_name"] = last_name
        if email is not None:
            user["email"] = email
        
        self._save_users()
        return True, "Profile updated successfully"
    
    def get_user_profile(self, username: str) -> Dict:
        """Get user profile information"""
        username_lower = username.lower()
        if username_lower not in self.users:
            return {}
        
        user = self.users[username_lower]
        return {
            "username": user.get("display_name", username),
            "first_name": user.get("first_name", ""),
            "last_name": user.get("last_name", ""),
            "email": user.get("email", ""),
            "role": user.get("role", "user"),
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login")
        }
    
    def authenticate(self, username: str, password: str) -> tuple:
        """Authenticate user with rate limiting - case insensitive username"""
        username_lower = username.lower()
        if username_lower not in self.users:
            return False, "Invalid username or password"
        
        user = self.users[username_lower]
        
        # Check if account is locked
        if user.get("locked_until") and time.time() < user["locked_until"]:
            remaining = int(user["locked_until"] - time.time())
            return False, f"Account locked. Try again in {remaining} seconds"
        
        # Check failed attempts
        if user.get("failed_attempts", 0) >= 5:
            # Lock account for 15 minutes
            user["locked_until"] = time.time() + (15 * 60)
            self._save_users()
            return False, "Too many failed attempts. Account locked for 15 minutes"
        
        # Verify password
        if self.verify_password(password, user["password_hash"], user["salt"]):
            # Reset failed attempts on successful login
            user["failed_attempts"] = 0
            user["last_login"] = time.time()
            user["locked_until"] = None
            self._save_users()
            return True, "Login successful"
        else:
            # Increment failed attempts
            user["failed_attempts"] = user.get("failed_attempts", 0) + 1
            self._save_users()
            return False, "Invalid username or password"
    
    def change_password(self, username: str, old_password: str, new_password: str) -> tuple:
        """Change user password - case insensitive username"""
        username_lower = username.lower()
        if username_lower not in self.users:
            return False, "User not found"
        
        user = self.users[username_lower]
        
        # Verify old password
        if not self.verify_password(old_password, user["password_hash"], user["salt"]):
            return False, "Current password is incorrect"
        
        # Validate new password
        if len(new_password) < 8:
            return False, "Password must be at least 8 characters long"
        
        # Hash new password
        password_hash, salt = self.hash_password(new_password)
        user["password_hash"] = password_hash
        user["salt"] = salt
        
        self._save_users()
        return True, "Password changed successfully"
    
    def get_user_role(self, username: str) -> Optional[str]:
        """Get user role - case insensitive username"""
        username_lower = username.lower()
        if username_lower in self.users:
            return self.users[username_lower].get("role", "user")
        return None
    
    def get_display_name(self, username: str) -> str:
        """Get the display name (original case) for a username"""
        username_lower = username.lower()
        if username_lower in self.users:
            return self.users[username_lower].get("display_name", username)
        return username
    
    def get_stats(self) -> dict:
        """Get user management statistics for debugging"""
        return {
            "total_users": len(self.users),
            "users_file_exists": self.users_file.exists(),
            "backup_file_exists": self.backup_file.exists(),
            "users_file_size": self.users_file.stat().st_size if self.users_file.exists() else 0,
            "last_modified": time.ctime(self.users_file.stat().st_mtime) if self.users_file.exists() else "Never"
        }

# Global user manager instance
user_manager = UserManager()

def check_password(username: str, password: str) -> tuple:
    """Check if username and password are correct"""
    return user_manager.authenticate(username, password)

def logout():
    """Clear session and logout"""
    # Log the logout
    if st.session_state.get('username'):
        st.info(f"User {st.session_state.username} logged out")
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()

def init_auth():
    """Initialize authentication state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0

def require_auth(required_role: str = None):
    """Decorator to require authentication for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated', False):
                st.error("Authentication required")
                return
            
            if required_role and st.session_state.get('user_role') != required_role:
                st.error("Insufficient permissions")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def is_admin() -> bool:
    """Check if current user is admin"""
    return st.session_state.get('user_role') == 'administrator'

def user_profile_form():
    """Display user profile editing form"""
    st.subheader("User Profile")
    
    # Get current profile
    profile = user_manager.get_user_profile(st.session_state.username)
    
    with st.form("user_profile"):
        first_name = st.text_input("First Name", value=profile.get("first_name", ""))
        last_name = st.text_input("Last Name", value=profile.get("last_name", ""))
        email = st.text_input("Email", value=profile.get("email", ""))
        
        # Show read-only fields
        st.text_input("Username", value=profile.get("username", ""), disabled=True)
        st.text_input("Role", value=profile.get("role", ""), disabled=True)
        
        if st.form_submit_button("Update Profile"):
            success, message = user_manager.update_user_profile(
                st.session_state.username,
                first_name=first_name,
                last_name=last_name,
                email=email
            )
            
            if success:
                st.success(message)
            else:
                st.error(message)

def change_password_form():
    """Display password change form"""
    st.subheader("Change Password")
    
    with st.form("change_password"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Change Password"):
            if new_password != confirm_password:
                st.error("New passwords do not match")
                return
            
            if len(new_password) < 8:
                st.error("Password must be at least 8 characters long")
                return
            
            success, message = user_manager.change_password(
                st.session_state.username, 
                current_password, 
                new_password
            )
            
            if success:
                st.success(message)
            else:
                st.error(message)

def admin_user_management():
    """Admin interface for user management"""
    if not is_admin():
        st.error("Admin access required")
        return
    
    st.subheader("User Management (Admin Only)")
    
    # Show user management stats
    stats = user_manager.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", stats["total_users"])
    with col2:
        st.metric("Users File Size", f"{stats['users_file_size']} bytes")
    with col3:
        st.write(f"**Last Modified:** {stats['last_modified']}")
    
    # Add new user
    with st.expander("Add New User"):
        with st.form("add_user"):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input("Username")
                new_first_name = st.text_input("First Name")
                new_email = st.text_input("Email")
            with col2:
                new_password = st.text_input("Password", type="password")
                new_last_name = st.text_input("Last Name")
                new_role = st.selectbox("Role", ["user", "administrator"])
            
            if st.form_submit_button("Add User"):
                if len(new_password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    success = user_manager.add_user(
                        new_username, new_password, new_role,
                        new_first_name, new_last_name, new_email
                    )
                    if success:
                        st.success(f"User {new_username} added successfully")
                    else:
                        st.error(f"User {new_username} already exists")
    
    # Display existing users
    st.write("**Existing Users:**")
    users_data = []
    for username, user_data in user_manager.users.items():
        display_name = user_data.get("display_name", username)
        first_name = user_data.get("first_name", "")
        last_name = user_data.get("last_name", "")
        full_name = f"{first_name} {last_name}".strip() or "Not provided"
        
        users_data.append({
            "Username": display_name,
            "Full Name": full_name,
            "Email": user_data.get("email", "Not provided"),
            "Role": user_data.get("role", "user"),
            "Last Login": time.ctime(user_data["last_login"]) if user_data.get("last_login") else "Never",
            "Failed Attempts": user_data.get("failed_attempts", 0)
        })
    
    if users_data:
        import pandas as pd
        df = pd.DataFrame(users_data)
        st.dataframe(df, width='stretch')