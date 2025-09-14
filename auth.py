import streamlit as st
from streamlit_oauth import OAuth2Component

CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = "https://mediwis.streamlit.app/"  # Your deployed Streamlit URI
oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    redirect_uri=REDIRECT_URI,
    scopes=["openid", "email", "profile"]
)

def login():
    result = oauth2.authorize_button("Continue with Google", "google")
    if result and "email" in result:
        st.session_state["user"] = {
            "name": result["name"],
            "email": result["email"],
            "picture": result["picture"]
        }
        st.experimental_rerun()
    elif "user" not in st.session_state:
        st.warning("Please sign in with Google to continue.")

def logout():
    if st.button("Logout"):
        st.session_state.pop("user", None)
        st.experimental_rerun()

def is_authenticated():
    return "user" in st.session_state

def get_user():
    return st.session_state.get("user", None)