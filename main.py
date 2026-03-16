import os
import sys
import subprocess

def main():
    print("Starting QFSVM Stock Market Prediction Dashboard...")
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'dashboard.py')
    
    # Use sys.executable so the venv Python (and its packages) are always used
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=False)

if __name__ == "__main__":
    main()
