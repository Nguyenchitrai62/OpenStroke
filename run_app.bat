@echo off
REM Mo app test model OpenStroke HVAC (http://127.0.0.1:8765).
REM ===== Doi model o day (dan duong dan file .pt, hoac keo tha .pt vao run_app.bat) =====
set "MODEL=F:\Downloads\OpenStroke40\e141_e122_ft_e13v2\fold0\best.pt"
REM ======================================================================================
if not "%~1"=="" set "MODEL=%~f1"
set "MODEL=%MODEL:"=%"
if "%MODEL%"=="" set "MODEL=%~dp0models\openstroke28.pt"

if not defined OPENSTROKE_EXTRACTOR_PYTHON set "OPENSTROKE_EXTRACTOR_PYTHON=F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe"
cd /d "%~dp0"
set "PY=py -3.12"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not exist "%MODEL%" (
  echo Khong thay model: %MODEL%
  echo Doi dong "set MODEL=" o dau file run_app.bat, hoac keo tha file .pt vao run_app.bat.
  pause
  exit /b 1
)
if not exist "%OPENSTROKE_EXTRACTOR_PYTHON%" (
  echo Khong thay Python cua AI_Takeoff: %OPENSTROKE_EXTRACTOR_PYTHON%
  pause
  exit /b 1
)
echo Model : %MODEL%
%PY% app\hvac_app.py --model "%MODEL%" --port 8765
pause
