@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ was not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
if not exist .venv (
  echo Creating private Python environment ^(first run only^)...
  python -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || (echo Package install failed - check your internet connection. & pause & exit /b 1)
echo Starting the Workbench - your browser will open. Close this window to stop it.
python -m streamlit run app.py
pause
