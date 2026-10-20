@echo off
REM Mo app test model OpenStroke (keo tha PDF). Doi model: sua MODEL_PATH trong app\streamlit_app.py hoac go trong panel.
cd /d "%~dp0"
set OMP_NUM_THREADS=1
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=py -3.12"
)
%PY% -c "import streamlit, torch, shapely" 2>nul
if errorlevel 1 (
  echo Thieu thu vien, dang cai streamlit...
  %PY% -m pip install -q streamlit
)
if not exist "F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe" if not defined OPENSTROKE_EXTRACTOR_PYTHON (
  echo Canh bao: khong thay venv AI_Takeoff de trich PDF. Dat OPENSTROKE_EXTRACTOR_PYTHON tro toi python co pdf_etl.
)
%PY% -m streamlit run app\streamlit_app.py
pause
