#!/usr/bin/env python3
"""
Budget Behavior Clustering - Project Organizer

This script reorganizes the project files into the proper structure:
- Creates necessary directories
- Moves Python modules to backend/
- Moves data files to backend/data/
- Moves React files to frontend/BudgetBehaviorClusteringReact/
- Creates necessary configuration files
"""

import os
import shutil
import sys

def create_directory_structure():
    """Create the necessary directory structure."""
    # Define base and target structure
    base_dir = os.getcwd()
    
    # Backend directories
    backend_dir = os.path.join(base_dir, "backend")
    backend_data_dir = os.path.join(backend_dir, "data")
    backend_models_dir = os.path.join(backend_dir, "models")
    backend_logs_dir = os.path.join(backend_dir, "logs")
    
    # Frontend directories
    frontend_dir = os.path.join(base_dir, "frontend")
    frontend_react_dir = os.path.join(frontend_dir, "BudgetBehaviorClusteringReact")
    frontend_public_dir = os.path.join(frontend_react_dir, "public")
    frontend_src_dir = os.path.join(frontend_react_dir, "src")
    frontend_components_dir = os.path.join(frontend_src_dir, "components")
    
    # Create directories
    directories = [
        backend_dir, 
        backend_data_dir,
        backend_models_dir,
        backend_logs_dir,
        frontend_dir,
        frontend_react_dir,
        frontend_public_dir,
        frontend_src_dir,
        frontend_components_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created")
    return (base_dir, backend_dir, backend_data_dir, frontend_react_dir, 
            frontend_public_dir, frontend_src_dir, frontend_components_dir)

def move_file(src_path, dst_path, file_type="file"):
    """Move a file and report results."""
    if os.path.exists(src_path):
        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Handle file already exists scenario
        if os.path.exists(dst_path):
            # If destination already exists, only copy if source is newer
            src_mtime = os.path.getmtime(src_path)
            dst_mtime = os.path.getmtime(dst_path)
            
            if src_mtime > dst_mtime:
                print(f"🔄 Replacing existing {file_type}: {os.path.basename(dst_path)}")
                shutil.copy2(src_path, dst_path)
            else:
                print(f"⏩ Skipping {file_type} (destination is newer): {os.path.basename(dst_path)}")
        else:
            # If destination doesn't exist, move or copy the file
            try:
                shutil.move(src_path, dst_path)
                print(f"📦 Moved {file_type}: {os.path.basename(src_path)} → {os.path.relpath(dst_path, os.getcwd())}")
            except (shutil.Error, OSError):
                # If move fails, try copy instead
                shutil.copy2(src_path, dst_path)
                print(f"📋 Copied {file_type}: {os.path.basename(src_path)} → {os.path.relpath(dst_path, os.getcwd())}")
        return True
    else:
        print(f"⚠️ {file_type.capitalize()} not found: {os.path.basename(src_path)}")
        return False

def organize_backend_files(base_dir, backend_dir):
    """Move Python modules to the backend directory."""
    backend_files = [
        "app.py",
        "cluster.py",
        "sample_data.py",
        "utils.py",
        "visuals.py",
        "budget_behavior_clustering.py"  # This might be renamed or not needed
    ]
    
    moved_count = 0
    for filename in backend_files:
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(backend_dir, filename)
        if move_file(src_path, dst_path, "module"):
            moved_count += 1
    
    # Check if we need to create __init__.py
    init_path = os.path.join(backend_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            f.write('"""Backend package for Budget Behavior Clustering."""\n')
        print(f"📝 Created {os.path.relpath(init_path, base_dir)}")
    
    return moved_count

def organize_data_files(base_dir, backend_data_dir):
    """Move data files to the backend/data directory."""
    # Data files might be in root or in a data folder
    data_source_dirs = [
        base_dir,
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "sample_data")
    ]
    
    data_files = [
        "Sample_Users_Budget_Profile.csv", 
        "Sample_Users_Budget_Profiles.csv",  # Handle potential pluralization
        "sample_transactions.csv",
        "budget_metrics.csv",
        "Transaction_Template.csv"
    ]
    
    moved_count = 0
    for filename in data_files:
        moved = False
        for source_dir in data_source_dirs:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(backend_data_dir, filename)
            
            if move_file(src_path, dst_path, "data file"):
                moved_count += 1
                moved = True
                break
        
        if not moved:
            print(f"⚠️ Data file not found in any location: {filename}")
    
    return moved_count

def organize_frontend_files(base_dir, frontend_react_dir, frontend_public_dir, frontend_src_dir):
    """Move React frontend files to the proper directories."""
    # Potential locations for React files
    react_source_dirs = [
        base_dir,
        os.path.join(base_dir, "BudgetBehaviorClusteringReact"),
        os.path.join(base_dir, "react"),
        os.path.join(base_dir, "frontend")
    ]
    
    # Public files (HTML, etc.)
    public_files = ["index.html", "favicon.ico", "manifest.json"]
    
    # Source files (JS, CSS)
    src_files = [
        "app.js", 
        "App.js",
        "index.js",
        "BudgetBehaviorClusteringApp.js"
    ]
    
    moved_count = 0
    
    # Move public files
    for filename in public_files:
        moved = False
        for source_dir in react_source_dirs:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(frontend_public_dir, filename)
            
            if move_file(src_path, dst_path, "frontend public file"):
                moved_count += 1
                moved = True
                break
    
    # Move source files
    for filename in src_files:
        moved = False
        for source_dir in react_source_dirs:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(frontend_src_dir, filename)
            
            if move_file(src_path, dst_path, "frontend source file"):
                moved_count += 1
                moved = True
                break
    
    return moved_count

def create_main_script(base_dir):
    """Create or update the main.py script."""
    main_path = os.path.join(base_dir, "main.py")
    
    # Check if main.py already exists
    if os.path.exists(main_path):
        print(f"ℹ️ main.py already exists at {os.path.relpath(main_path, base_dir)}")
        return
    
    # Basic main.py template
    main_content = '''"""
Budget Behavior Clustering - Main Application

This is the main entry point for the Budget Behavior Clustering application.
It sets up the required data and runs the Streamlit app.
"""

import os
import subprocess
import argparse
from pathlib import Path
import pandas as pd

# Import the modules
from backend.sample_data import save_sample_data, generate_sample_data
from backend.utils import setup_environment, initialize_logging
from backend.cluster import load_model, train_clustering_model

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "backend/data",
        "backend/models",
        "backend/logs",
        "frontend"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure initialized")

def check_data_files():
    """Check if required data files exist, generate them if not."""
    data_file = Path("backend/data/Sample_Users_Budget_Profile.csv")
    transactions_file = Path("backend/data/sample_transactions.csv")
    metrics_file = Path("backend/data/budget_metrics.csv")
    
    files_missing = not (data_file.exists() and transactions_file.exists() and metrics_file.exists())
    
    if files_missing:
        print("🔄 Generating sample data files...")
        save_sample_data(output_dir="backend/data")
    else:
        print("✅ Data files found")
    
    return data_file

def train_or_load_model(data_file):
    """Train a new clustering model or load an existing one."""
    model_file = Path("backend/models/kmeans_model.pkl")
    
    if not model_file.exists():
        print("🔄 Training new clustering model...")
        # Load data
        df = pd.read_csv(data_file)
        # Train model
        train_clustering_model(df, model_file)
        print("✅ Model trained and saved")
    else:
        print("✅ Using existing model")
        
    return model_file

def run_streamlit_app():
    """Run the Streamlit application."""
    app_path = Path("backend/app.py")
    
    if not app_path.exists():
        print("❌ Error: app.py not found")
        return False
    
    print("🚀 Launching Streamlit app...")
    # Run Streamlit in a subprocess
    process = subprocess.Popen(
        ["streamlit", "run", str(app_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print the URL where the app is running
    for line in process.stdout:
        print(line, end="")
        if "You can now view your Streamlit app in your browser" in line:
            break
    
    return process

def main():
    """Main function to run the Budget Behavior Clustering application."""
    parser = argparse.ArgumentParser(description="Budget Behavior Clustering Application")
    parser.add_argument("--regenerate-data", action="store_true", 
                        help="Force regeneration of sample data")
    parser.add_argument("--retrain-model", action="store_true", 
                        help="Force retraining of clustering model")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = initialize_logging(level=args.log_level)
    logger.info("Starting Budget Behavior Clustering application")
    
    # Setup the environment
    setup_environment()
    
    # Setup directories
    setup_directories()
    
    # Check and generate data if needed
    if args.regenerate_data:
        print("🔄 Regenerating sample data files...")
        save_sample_data(output_dir="backend/data")
        data_file = Path("backend/data/Sample_Users_Budget_Profile.csv")
    else:
        data_file = check_data_files()
    
    # Train or load model
    if args.retrain_model:
        print("🔄 Retraining clustering model...")
        df = pd.read_csv(data_file)
        model_file = Path("backend/models/kmeans_model.pkl")
        train_clustering_model(df, model_file)
        print("✅ Model retrained and saved")
    else:
        model_file = train_or_load_model(data_file)
    
    # Run the Streamlit app
    process = run_streamlit_app()
    
    if process:
        try:
            # Keep the script running until interrupted
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down the application...")
            process.terminate()
            process.wait()
            print("✅ Application shutdown complete")
    
    logger.info("Budget Behavior Clustering application stopped")

if __name__ == "__main__":
    main()
'''
    
    with open(main_path, 'w') as f:
        f.write(main_content)
    
    print(f"📝 Created main script: {os.path.relpath(main_path, base_dir)}")

def create_requirements_file(base_dir):
    """Create or update the requirements.txt file."""
    req_path = os.path.join(base_dir, "requirements.txt")
    
    requirements = [
        "streamlit==1.30.0",
        "pandas==2.1.4",
        "numpy==1.26.3",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "plotly==5.18.0",
        "seaborn==0.13.1",
        "scipy==1.12.0",
        "joblib==1.3.2"
    ]
    
    # Check if requirements.txt already exists
    if os.path.exists(req_path):
        print(f"ℹ️ requirements.txt already exists at {os.path.relpath(req_path, base_dir)}")
        return
    
    with open(req_path, 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"📝 Created requirements file: {os.path.relpath(req_path, base_dir)}")

def create_readme_file(base_dir):
    """Create a README.md file if it doesn't exist."""
    readme_path = os.path.join(base_dir, "README.md")
    
    # Check if README.md already exists
    if os.path.exists(readme_path):
        print(f"ℹ️ README.md already exists at {os.path.relpath(readme_path, base_dir)}")
        return
    
    readme_content = '''# Budget Behavior Clustering

A financial behavior analysis application that uses machine learning to cluster users into financial personas based on their budget data.

## Features

- User budget data input via Streamlit UI
- KMeans clustering for financial persona identification
- Visual comparisons (bar charts, radar charts, 3D visualizations)
- Personalized feedback and recommendations
- Financial health scoring

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

## Project Structure

```
BudgetBehaviorClustering/
├── backend/              # Python backend modules
│   ├── app.py            # Streamlit application interface
│   ├── cluster.py        # Clustering logic and model handling
│   ├── sample_data.py    # Sample data generation
│   ├── utils.py          # Utility functions
│   ├── visuals.py        # Visualization functions
│   ├── data/             # CSV data files
│   ├── models/           # Saved clustering models
│   └── logs/             # Application logs
├── frontend/             # Optional React frontend
│   └── BudgetBehaviorClusteringReact/
│       ├── public/
│       └── src/
├── main.py               # Main application entry point
└── requirements.txt      # Python dependencies
```

## Usage

Run the application with:

```bash
python main.py
```

For more options:

```bash
python main.py --help
```
'''
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created README file: {os.path.relpath(readme_path, base_dir)}")

def main():
    """Main function to organize the project."""
    print("🔍 Budget Behavior Clustering Project Organizer")
    print("==============================================")
    
    # Create directory structure
    (base_dir, backend_dir, backend_data_dir, frontend_react_dir, 
     frontend_public_dir, frontend_src_dir, frontend_components_dir) = create_directory_structure()
    
    # Organize the files
    backend_count = organize_backend_files(base_dir, backend_dir)
    data_count = organize_data_files(base_dir, backend_data_dir)
    frontend_count = organize_frontend_files(base_dir, frontend_react_dir, frontend_public_dir, frontend_src_dir)
    
    # Create auxiliary files
    create_main_script(base_dir)
    create_requirements_file(base_dir)
    create_readme_file(base_dir)
    
    # Summary
    print("\n📊 Organization Summary")
    print("==============================================")
    print(f"Backend modules organized: {backend_count}")
    print(f"Data files organized: {data_count}")
    print(f"Frontend files organized: {frontend_count}")
    print("\n🎉 Project successfully reorganized for Budget Behavior Clustering!")
    print("Run 'python main.py' to start the application")

if __name__ == "__main__":
    main()