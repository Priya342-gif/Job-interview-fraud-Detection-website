import pytesseract
import sys
from PIL import Image

# Try to find Tesseract
tesseract_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]

found = False
for path in tesseract_paths:
    import os
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        found = True
        print(f'✓ Tesseract found at: {path}')
        try:
            version = pytesseract.get_tesseract_version()
            print(f'✓ Version: {version}')
            print('✓ Ready to use!')
        except:
            print('✗ Found but cannot run')
        break

if not found:
    print('✗ Tesseract not installed')
    print('Please install from: https://github.com/UB-Mannheim/tesseract/wiki')
    sys.exit(1)
