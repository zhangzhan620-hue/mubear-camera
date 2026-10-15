@echo off
venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --name mubear ^
  --add-data "app/static;app/static" ^
  --collect-all mediapipe ^
  --collect-all onnxruntime ^
  --collect-all pyttsx3 ^
  --hidden-import pyttsx3.drivers.sapi5 ^
  --exclude-module torch --exclude-module torchvision --exclude-module ultralytics ^
  --exclude-module jax --exclude-module jaxlib ^
  --exclude-module polars --exclude-module _polars_runtime_32 ^
  --exclude-module onnx --exclude-module onnxslim ^
  --exclude-module scipy --exclude-module pandas ^
  --exclude-module matplotlib --exclude-module seaborn ^
  --exclude-module tensorboard --exclude-module IPython --exclude-module notebook ^
  run.py
echo.
echo Build done: dist\mubear\mubear.exe
pause
