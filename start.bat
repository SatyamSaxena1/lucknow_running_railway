@echo off

call C:\ProgramData\anaconda3\condabin\activate.bat wagon

@REM python ./testing_only/hello_there.py
start /B python driver_launcher.py
python main_launcher.py
