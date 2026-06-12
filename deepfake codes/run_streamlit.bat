@echo off
cd /d "%~dp0"
".\venv\Scripts\python.exe" -m streamlit run "frontend\streamlit_app.py"
pause
