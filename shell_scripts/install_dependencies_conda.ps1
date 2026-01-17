# Windows PowerShell installation script for WAVES using Conda
# Please run this script in the root directory of this repo: .\shell_scripts\install_dependencies_conda.ps1

$envName = "waves"
$pythonVersion = "3.10"

Write-Host "Setting up WAVES environment for Windows using Conda (Python $pythonVersion)..." -ForegroundColor Cyan

# 1. Check if conda is available
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda not found! Please install Anaconda or Miniconda first." -ForegroundColor Red
    exit
}

# 2. Create Conda Environment
Write-Host "Creating conda environment '$envName' with Python $pythonVersion..." -ForegroundColor Cyan
conda create -n $envName python=$pythonVersion -y

# 3. Activate Environment
# Note: For script execution, we use 'conda run' or instruct user to activate
Write-Host "Installing dependencies into '$envName'..." -ForegroundColor Cyan

# Use conda run to ensure we are installing into the right environment without needing full shell activation in script
function Conda-Pip-Install {
    param([string]$packages, [string]$extraArgs = "")
    Write-Host "Installing: $packages $extraArgs" -ForegroundColor Gray
    if ($extraArgs -ne "") {
        conda run -n $envName pip install $extraArgs $packages.Split(" ")
    } else {
        conda run -n $envName pip install $packages.Split(" ")
    }
}

# 4. Basic Tools
Conda-Pip-Install "--upgrade pip ipython jupyter ipywidgets python-dotenv"

# 5. PyTorch (CUDA 11.8)
Write-Host "Installing PyTorch 2.1.0 (CUDA 11.8)..." -ForegroundColor Cyan
conda run -n $envName pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. xformers
Write-Host "Installing xformers..." -ForegroundColor Cyan
conda run -n $envName pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# 7. ML & Data Processing
Conda-Pip-Install "transformers==4.35.2 diffusers datasets[vision] ftfy"
Conda-Pip-Install "onnx onnxruntime-gpu torchmetrics open_clip_torch torchattacks scikit-learn scikit-image pandas matplotlib imageio opencv-python pycocotools"
Conda-Pip-Install "omegaconf einops timm lpips"

# 8. CLIP and other GitHub repos
Write-Host "Installing CLIP from GitHub..." -ForegroundColor Cyan
conda run -n $envName pip install git+https://github.com/openai/CLIP.git

# 9. Parallel & Web
Conda-Pip-Install "accelerate"
Conda-Pip-Install "huggingface-hub gitpython gradio==4.3.0 plotly plotly-express wordcloud"

# NOTE: deepspeed on Windows
Write-Host "Attempting to install deepspeed (likely to fail on Windows)..." -ForegroundColor Yellow
conda run -n $envName pip install deepspeed

# 10. Final fix
Conda-Pip-Install "ipython==8.16.1"

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "To start working, run:" -ForegroundColor Green
Write-Host "conda activate $envName" -ForegroundColor Yellow
