@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m streamlit run app\app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
) else (
    python -m streamlit run app\app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
)
