# Windows PowerShell installation script for WAVES
# Please run this script in the root directory of this repo: .\shell_scripts\install_dependencies.ps1

Write-Host "Setting up WAVES environment for Windows..." -ForegroundColor Cyan

# 1. Create Virtual Environment
if (!(Test-Path -Path ".\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists, skipping creation." -ForegroundColor Gray
}

# 2. Activate Virtual Environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# 3. Upgrade Pip and install basic tools
Write-Host "Upgrading pip and installing basic tools..." -ForegroundColor Cyan
python -m pip install --upgrade pip ipython jupyter ipywidgets python-dotenv

# 4. Install PyTorch with CUDA 11.8 support
Write-Host "Installing PyTorch 2.1.0 (CUDA 11.8)..." -ForegroundColor Cyan
python -m pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install Huggingface and related libraries
Write-Host "Installing Huggingface libraries..." -ForegroundColor Cyan
python -m pip install transformers diffusers "datasets[vision]" ftfy

# 6. Install xformers
Write-Host "Installing xformers..." -ForegroundColor Cyan
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# 7. Install other ML and Data libraries
Write-Host "Installing ML and Data processing libraries..." -ForegroundColor Cyan
# pycocotools might need C++ Build Tools. If it fails, try 'pip install pycocotools-windows'
python -m pip install onnx onnxruntime-gpu torchmetrics open_clip_torch torchattacks scikit-learn scikit-image pandas matplotlib imageio opencv-python pycocotools

# 8. Install Metric libraries
Write-Host "Installing CLIP from GitHub..." -ForegroundColor Cyan
python -m pip install git+https://github.com/openai/CLIP.git

# 9. Install Parallel and web libraries
Write-Host "Installing Parallel and Web libraries..." -ForegroundColor Cyan
python -m pip install accelerate

# NOTE: deepspeed is difficult to install on Windows. 
# Many features in this repo may work without it.
Write-Host "Attempting to install deepspeed (this often fails on Windows without Visual Studio and Ninja)..." -ForegroundColor Yellow
python -m pip install deepspeed

python -m pip install huggingface-hub gitpython gradio==4.3.0 plotly plotly-express wordcloud

# 10. Final fixes
Write-Host "Applying Jupyter fix..." -ForegroundColor Cyan
python -m pip install ipython==8.16.1

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "To start working, run:" -ForegroundColor Green
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
