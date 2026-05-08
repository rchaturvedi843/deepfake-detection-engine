# Deepfake Detection Engine

## Overview
Deepfake Detection Engine is an advanced machine learning project built with PyTorch and OpenCV to detect artificially generated or manipulated media (deepfakes) in images and videos. The system uses a Streamlit-based web interface for easy uploading and analysis, providing users with a comprehensive report of its findings.

## Features
- **Video & Image Analysis**: Processes both video files and static images to detect manipulation.
- **Deep Learning Powered**: Utilizes custom PyTorch models (`deepfake_final_model.pth`, `deepfake_strong_model.pth`, etc.) trained on robust datasets to accurately identify deepfake artifacts.
- **Face Detection**: Uses MediaPipe and OpenCV to isolate and analyze faces from frames.
- **Interactive UI**: An intuitive Streamlit frontend (`frontend/streamlit_app.py`) for seamless user interaction.
- **Automated Reporting**: Generates downloadable PDF reports detailing the confidence scores and analysis results using ReportLab.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rchaturvedi843/deepfake-detection-engine.git
   cd deepfake-detection-engine
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   cd "deepfake codes"
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface (Streamlit)
To launch the user interface locally, run the provided batch script or use the Streamlit CLI:

**Option 1: Using the Batch Script (Windows)**
Simply double-click `run_streamlit.bat` located inside the `deepfake codes` folder.

**Option 2: Using the Command Line**
```bash
cd "deepfake codes"
streamlit run frontend/streamlit_app.py
```

## Directory Structure
- `deepfake codes/models/`: Contains the PyTorch models (`.pth` files) for deepfake detection.
- `deepfake codes/frontend/`: Contains the Streamlit app code (`streamlit_app.py`).
- `deepfake codes/preprocessing/`: Scripts to prepare and organize datasets for training.
- `deepfake codes/utils/`: Utility scripts for generating reports and managing cache.
- `deepfake codes/notebooks/`: Jupyter Notebooks used for model exploration and data analysis.

## Requirements
- Python 3.8+
- PyTorch
- Streamlit
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy, Pandas, Scikit-learn
- ReportLab

## License
This project is open-source. Please see the `LICENSE` file for more details.
