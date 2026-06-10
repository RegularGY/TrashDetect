@echo off
title EcoSort AI — Waste Detection System

echo.
echo  =============================================
echo   EcoSort AI — Waste Detection System
echo   Starting server...
echo  =============================================
echo.

:: Activate virtual environment
call C:\Users\Victus\Documents\GitHub\TrashDetect\StaticDetectCode\ecosort-env\Scripts\activate.bat

:: Start Flask in background and open browser after 2 seconds
start /b python C:\Users\Victus\Documents\GitHub\TrashDetect\StaticDetectCode\app.py

:: Wait 2 seconds for Flask to start then open browser
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5000

echo  Server running at http://127.0.0.1:5000
echo  Close this window to stop the server.
echo.

:: Keep window open so Flask keeps running
pause >nul
