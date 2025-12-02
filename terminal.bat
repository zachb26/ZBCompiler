@echo off
REM This command opens a new Command Prompt window and executes the CD command
REM to change the directory to the specified path.
start cmd /k "cd /d C:\brazingtoncompiler"
REM The "/d" flag ensures the command works even if the directory was on a different drive.
exit