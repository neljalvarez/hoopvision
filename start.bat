@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set PY=.venv\Scripts\python.exe
if not defined CAM set CAM=0
:menu
echo.
echo  HoopVision   (camera %CAM%)
echo  ----------
echo  1  Start tracking
echo  2  Start tracking with the debug view
echo  3  Calibrate the hoop
echo  4  Calibrate the ball
echo  5  Analyse a video file
echo  6  Validation: label a test video
echo  7  Validation: evaluate a labelled video
echo  8  Use a different camera number
echo  Q  Quit
set CH=
set /p CH=Choose: 
if /i "%CH%"=="1" goto run
if /i "%CH%"=="2" goto debug
if /i "%CH%"=="3" goto calhoop
if /i "%CH%"=="4" goto calball
if /i "%CH%"=="5" goto video
if /i "%CH%"=="6" goto label
if /i "%CH%"=="7" goto evaluate
if /i "%CH%"=="8" goto camera
if /i "%CH%"=="q" exit /b 0
goto menu

:run
%PY% main.py --camera %CAM%
goto menu
:debug
%PY% main.py --camera %CAM% --debug
goto menu
:calhoop
%PY% main.py --camera %CAM% --calibrate
goto menu
:calball
%PY% main.py --camera %CAM% --calibrate-ball
goto menu
:video
set /p VID=Path to the video file (you can drag it into this window): 
%PY% main.py --video %VID% --debug
goto menu
:label
set /p VID=Path to the video file: 
%PY% tools\validate.py label %VID%
goto menu
:evaluate
set /p VID=Path to the video file: 
%PY% tools\validate.py evaluate %VID%
goto menu
:camera
set /p CAM=Camera number (0, 1, 2...): 
goto menu
