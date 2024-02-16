@echo off
:: Activate the TensorFlow environment. Modify the path below to point to your own environment activation script.
CALL "C:\Path\To\Your\Environment\Scripts\activate"

:: Setup your CUDA and cuDNN environment. These paths need to be adjusted to where CUDA and cuDNN are installed on your system.
SET PATH=C:\Path\To\CUDA\Library\bin;C:\Path\To\cuDNN\Library\bin;%PATH%

:: Execute your Python script. Change the following path to where your Python script is located.
python "C:\Path\To\Your\Script\cats_training.py"

:: Pause the script execution to see any messages or errors.
pause
