$ErrorActionPreference = 'Stop'

# cd to repo root (parent of scripts)
Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..

Write-Host "[run_tests] Python: $(python --version)"
Write-Host "[run_tests] PyTorch installed: $(python -c \"import importlib,sys;print('torch' if importlib.util.find_spec('torch') else 'NOT INSTALLED')\")"
Write-Host "[run_tests] CUDA available: $(python -c \"import torch,sys;print(torch.cuda.is_available())\")"
if (python - << 'PY'
import os, torch
if torch.cuda.is_available():
    print('[run_tests] CUDA device:', torch.cuda.get_device_name(0))
else:
    print('[run_tests] CUDA device: none')
PY
) { }

Write-Host "[run_tests] Running pytest..."
python -m pytest

if ($LASTEXITCODE -eq 0) {
    Write-Host "[run_tests] SUCCESS: All tests passed." -ForegroundColor Green
} else {
    Write-Host "[run_tests] FAILURE: Some tests failed (exit $LASTEXITCODE)." -ForegroundColor Red
}

