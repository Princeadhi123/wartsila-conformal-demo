#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
else
    PY="python"
fi
"$PY" -m streamlit run app/app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
