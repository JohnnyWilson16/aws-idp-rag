"""
Streamlit Community Cloud Entrypoint for AWS Serverless IDP + RAG Assistant.
Loads and executes frontend/app.py.
"""
import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Run frontend application
target_script = os.path.join(BASE_DIR, "frontend", "app.py")
with open(target_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), target_script, "exec")
    exec(code, globals())
