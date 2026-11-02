@echo off
rem Starts the AI461 palmprint demo on the Tongji database (all 600 palms).
rem The first start on new code encodes all 12,000 images and scores them,
rem which takes several minutes; every start after that loads the saved
rem templates and scores in a few seconds.
cd /d "%~dp0"
python demo.py --data data\tongji\ROI %*
if errorlevel 1 (
  echo.
  echo The demo stopped with an error. Details, if any, are in demo_error.log
  pause
)
