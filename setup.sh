---

## ⚡ `setup.sh`

```bash
#!/bin/bash
# Setup script for ASL Sign Detection Web App
# Run this in your VS Code terminal: bash setup.sh

echo "🔹 Creating Python virtual environment..."
python3 -m venv venv

echo "🔹 Activating virtual environment..."
source venv/bin/activate

echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing Python dependencies (training + testing)..."
pip install \
  tensorflow==2.12.0 \
  keras \
  opencv-python \
  mediapipe \
  jupyter \
  numpy \
  matplotlib

echo "🔹 Installing TensorFlow.js converter (to export models for browser)..."
pip install tensorflowjs

echo "✅ All dependencies installed!"
echo "👉 To activate environment again later: source venv/bin/activate"
echo "👉 To run local server for testing frontend: python3 -m http.server 8000"
