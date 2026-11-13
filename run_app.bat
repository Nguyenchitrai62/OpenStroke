@echo off
REM ================================================================================================
REM  OpenStroke HVAC - thu model chon net (OpenStroke28) tren PDF ban ve HVAC.
REM  Cach dung: bam dup file nay, hoac keo tha file .pt vao file nay de dung model khac.
REM  Trinh duyet tu mo http://127.0.0.1:8765  (dong cua so den nay de tat app).
REM
REM  Cac buoc cho moi trang:
REM    1. Trich PDF                      (Python cua AI_Takeoff, pdf_etl)
REM    2. Tien xu ly (giong luc train)   (moi net -> hinh dang, mau, do day, net xung quanh; khong doc layer)
REM    3. Chay model                     (xac suat tung net la net can boc)
REM  Nguong, mau, thong ke tinh ngay tren trinh duyet. "Chay logic" = logic HVAC cua AI_Takeoff
REM  tren cac net >= nguong (giu layer that, hoac coi tat ca la mot layer).
REM ================================================================================================

REM ----- Model: doi duong dan o day ----------------------------------------------------------------
set "MODEL=%~dp0models\openstroke28.pt"
if not exist "%MODEL%" if exist "F:\Downloads\v28.pt" set "MODEL=F:\Downloads\v28.pt"
if not "%~1"=="" set "MODEL=%~f1"

REM ----- Python cua AI_Takeoff (de trich PDF va chay logic HVAC) -----------------------------------
if not defined OPENSTROKE_EXTRACTOR_PYTHON set "OPENSTROKE_EXTRACTOR_PYTHON=F:\Source_code\AI_Takeoff\BE\.venv\Scripts\python.exe"

cd /d "%~dp0"
set "PY=py -3.12"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not exist "%MODEL%" (
  echo Khong thay model: %MODEL%
  echo Sua dong "set MODEL=" trong run_app.bat, hoac keo tha file .pt vao run_app.bat.
  pause
  exit /b 1
)
if not exist "%OPENSTROKE_EXTRACTOR_PYTHON%" (
  echo Khong thay Python cua AI_Takeoff: %OPENSTROKE_EXTRACTOR_PYTHON%
  echo Dat bien OPENSTROKE_EXTRACTOR_PYTHON tro toi python co pdf_etl.
  pause
  exit /b 1
)

echo Model : %MODEL%
echo Dang mo app... (Ctrl+C hoac dong cua so nay de tat)
%PY% app\hvac_app.py --model "%MODEL%" --port 8765
pause
