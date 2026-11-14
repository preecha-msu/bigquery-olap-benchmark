@echo off
chcp 65001 >nul
cd /d "%~dp0"
python "scripts\hub_sync_notebook1_master.py" --mode push
pause
