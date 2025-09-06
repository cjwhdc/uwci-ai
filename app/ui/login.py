import streamlit as st
from app.auth import check_password, user_manager

def show_login_page():
    """Display login page"""
    st.title("🔒 UWCI Sermon AI - Login")
    st.markdown("Please login to access the sermon analysis system")
    
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    success, message = check_password(username, password)
                    if success:
                        st.session_state.authenticated = True
                        # Store the display name instead of lowercase username
                        st.session_state.username = user_manager.get_display_name(username)
                        st.session_state.user_role = user_manager.get_user_role(username)
                        
                        # Import here to avoid circular imports
                        from main import create_persistent_login
                        create_persistent_login(username)
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both username and password")
    
    # Show setup instructions if initial admin file exists
    import os
    initial_password_file = "app/data/initial_admin_password.txt"
    if os.path.exists(initial_password_file):
        with st.expander("First Time Setup", expanded=True):
            st.warning("**First time setup detected!**")
            st.info("Check the file `app/data/initial_admin_password.txt` for your initial admin credentials.")
            st.write("After logging in:")
            st.write("1. Change your admin password immediately")
            st.write("2. Create additional user accounts")
            st.write("3. Delete the initial password file")