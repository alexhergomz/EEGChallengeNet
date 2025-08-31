@echo off
setlocal
cd /d "%~dp0.."
echo [run_tests] Python:
python --version
echo [run_tests] PyTorch installed:
python -c "import importlib;print('torch' if importlib.util.find_spec('torch') else 'NOT INSTALLED')"
echo [run_tests] CUDA available:
python -c "import torch;print(torch.cuda.is_available())"
echo [run_tests] Running pytest...
python -m pytest
if %errorlevel%==0 (
  echo [run_tests] SUCCESS: All tests passed.
) else (
  echo [run_tests] FAILURE: Some tests failed (exit %errorlevel%).
)
