@echo off
:: Activate your TensorFlow environment. Replace the path below with the path to your own environment.
CALL "C:\Path\To\Your\Environment\Scripts\activate"

:: Set up your CUDA and cuDNN paths. These paths depend on where CUDA and cuDNN are installed on your system.
:: Replace these with the appropriate directories on your machine.
SET PATH=C:\Path\To\CUDA\Library\bin;C:\Path\To\cuDNN\Library\bin;%PATH%

:: Run your Python script. Replace the path below with the path to your script.
python "C:\Path\To\Your\Script\human_model_tester_game.py"

:: Pause the script to view any output or errors.
pause
