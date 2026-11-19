@echo off
cd /d "%~dp0"
echo Starting BLP Field Reporting... close this window to stop it.
python -m streamlit run app.py
pause
