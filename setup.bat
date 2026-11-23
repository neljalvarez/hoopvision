@echo off
REM HoopVision one-time setup: creates a private Python environment in .venv and installs the packages.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation, then run setup.bat again.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo Package installation failed. & pause & exit /b 1)
python -c "import cv2, numpy; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__, '- OK')"
echo.
echo Setup complete. Run start.bat
pause
