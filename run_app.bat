@echo off
REM Ensure script runs from the project directory where this .bat lives
cd /d "%~dp0"

REM Activate venv (cmd activation)
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found at .venv\Scripts\activate.bat
    echo Create it with: python -m venv .venv
    echo Then run: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Launch the modular app (the older single-file SmartBizLQ_streamlit_app.py is kept for reference only)
".venv\Scripts\python.exe" -m streamlit run "streamlit_app.py"

REM Pause so you can read any error output
pause
