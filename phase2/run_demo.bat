@echo off
rem Starts the AI461 palmprint verification demo. Double-click this file.
rem The first start generates the synthetic demo data (about half a minute);
rem every start after that takes a second or two.
cd /d "%~dp0"
python demo.py %*
if errorlevel 1 (
  echo.
  echo The demo stopped with an error. Details, if any, are in demo_error.log
  pause
)
