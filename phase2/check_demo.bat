@echo off
rem Rehearses every scenario of the live demo on THIS laptop: steps through
rem each one by itself, saves a screenshot of the real window for each, and
rem writes a pass/fail report. Takes about a minute; do not touch the mouse
rem or keyboard while it runs.
cd /d "%~dp0"
python demo.py --check
echo.
echo Report:      _screens\check_report.txt
echo Screenshots: _screens\
pause
