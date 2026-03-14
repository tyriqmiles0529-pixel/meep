import subprocess
import sys
import os

# PHASE V5: High-Performance Stability Fix (Windows MKL)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def launch():
    print("="*60)
    print("MEEP TERMINAL | Commercial Engineering Environment")
    print("="*60)
    print("Starting Streamlit UI Layer...")
    
    # Check for requirements
    try:
        import streamlit
        import sqlalchemy
    except ImportError:
        print("[!] Missing dependencies. Running pip install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "sqlalchemy", "plotly"])

    # Launch Streamlit with the explicit Python interpreter to ensure local pathing
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--theme.base", "dark"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTerminal Shutdown Complete.")

if __name__ == "__main__":
    launch()
