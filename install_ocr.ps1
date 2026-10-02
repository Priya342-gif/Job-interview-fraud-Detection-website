# Fraud Detection System - OCR Setup Script
# This script installs Tesseract OCR automatically

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fraud Detection - OCR Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python packages are installed
Write-Host "[1/3] Checking Python packages..." -ForegroundColor Yellow
try {
    python -c "import pytesseract" 2>$null
    Write-Host "  ✓ pytesseract already installed" -ForegroundColor Green
} catch {
    Write-Host "  Installing pytesseract..." -ForegroundColor Yellow
    pip install pytesseract Pillow
}

# Check if Tesseract is installed
Write-Host ""
Write-Host "[2/3] Checking Tesseract OCR..." -ForegroundColor Yellow

$tesseractPaths = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
)

$tesseractFound = $false
foreach ($path in $tesseractPaths) {
    if (Test-Path $path) {
        Write-Host "  ✓ Tesseract found at: $path" -ForegroundColor Green
        $tesseractFound = $true
        break
    }
}

if (-not $tesseractFound) {
    Write-Host "  ✗ Tesseract not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please install Tesseract manually:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor White
    Write-Host "  2. Download latest version (tesseract-ocr-w64-setup-5.x.x.exe)" -ForegroundColor White
    Write-Host "  3. Run installer with default settings" -ForegroundColor White
    Write-Host "  4. Run this script again" -ForegroundColor White
    Write-Host ""
    
    # Try to open browser
    $response = Read-Host "Open download page in browser? (y/n)"
    if ($response -eq 'y') {
        Start-Process "https://github.com/UB-Mannheim/tesseract/wiki"
    }
}

# Test installation
Write-Host ""
Write-Host "[3/3] Testing installation..." -ForegroundColor Yellow

if ($tesseractFound) {
    Write-Host "  ✓ All requirements met!" -ForegroundColor Green
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "✓ Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run the app with:" -ForegroundColor Yellow
    Write-Host "  streamlit run app_with_image.py" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "⚠ Setup Incomplete" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Please install Tesseract OCR manually and run this script again." -ForegroundColor Yellow
    Write-Host ""
}

Read-Host "Press Enter to exit"
