@echo off
title DocuMind AI Launcher
echo ============================================================
echo   Starting DocuMind AI — Production RAG Knowledge Assistant
echo ============================================================
echo.

IF EXIST venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python run_app.py
pause
