@echo off
REM Mo app test model OpenStroke (keo tha PDF).
REM Model tot nhat hien tai: E54 fold 3. Hau ky tu doc runs_colab\e54\postprocess.json.
REM ===== Doi model o day (duong dan .pt). De trong thi dung MODEL_PATH trong app\streamlit_app.py =====
set MODEL="F:\Downloads\openstroke9.pt"
REM Hoac truyen duong dan khi chay:  run_app.bat "F:\Downloads\model.pt"   (hoac keo tha file .pt vao run_app.bat)
if not "%~1"=="" set MODEL="%~f1"
REM ====================================================================================================
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set "OPENSTROKE_MODEL=%MODEL:"=%"
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
if defined MODEL if not exist %MODEL% echo Canh bao: khong thay model %MODEL%
echo Model: %MODEL%
%PY% -m streamlit run app\streamlit_app.py
pause
