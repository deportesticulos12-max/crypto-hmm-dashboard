@echo off
echo Iniciando Crypto-HMM Dashboard...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run app.py
pause
