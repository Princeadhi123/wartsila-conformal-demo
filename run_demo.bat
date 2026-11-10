@echo off
cd /d "%~dp0"
python -m streamlit run app\app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
