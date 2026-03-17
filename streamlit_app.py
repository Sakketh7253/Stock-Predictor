"""Streamlit entrypoint for Streamlit Cloud.

Place this file at the repository root and set it as the app's "Main file" on
Streamlit Cloud (streamlit_app.py). It simply imports the dashboard module which
builds the Streamlit UI at import time.
"""

import app.dashboard  # noqa: F401  # module import triggers the Streamlit app
