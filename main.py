#!/usr/bin/env python3
"""
Budget Behavior Clustering - Main Application

This is the main entry point for the Budget Behavior Clustering application.
It sets up the required data and runs the Streamlit app.
"""

import os
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import sys

# Add the project directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import the modules
try:
    from backend.sample_data import save_sample_data, generate_sample_data
    from backend.utils import setup_environment, initialize_logging
    from backend.cluster import load_model, train_clustering_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure your project structure is correct and modules are available.")
    sys.exit(1)

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

def train_model():
    """Train or load the clustering model."""
    model_path = "backend/models/budget_cluster_model.joblib"
    data_path = "backend/data/Sample_Users_Budget_Profile.csv"
    
    # Check if model exists
    if os.path.exists(model_path):
        print("✅ Clustering model found")
        return
    
    # Check if data exists
    if not os.path.exists(data_path):
        print("❌ Cannot train model: data file not found")
        return
    
    # Load data and train model
    try:
        print("🔄 Training clustering model...")
        df = pd.read_csv(data_path)
        
        # Train model and save
        train_clustering_model(df, model_path)
        print("✅ Clustering model trained and saved")
    except Exception as e:
        print(f"❌ Error training model: {e}")

def run_streamlit_app():
    """Run the Streamlit application."""
    app_path = os.path.join(BASE_DIR, "backend", "app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Streamlit app file not found: {app_path}")
        return False
    
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        return False

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Budget Behavior Clustering Application")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample data")
    parser.add_argument("--train-model", action="store_true", help="Train clustering model")
    parser.add_argument("--no-run", action="store_true", help="Don't run the Streamlit app")
    args = parser.parse_args()
    
    # Initialize logging
    logger = initialize_logging()
    logger.info("Application starting")
    
    # Setup environment
    setup_environment()
    
    # Setup directories
    setup_directories()
    
    # Generate data if requested or needed
    if args.generate_data:
        save_sample_data(output_dir="backend/data")
    else:
        check_data_files()
    
    # Train model if requested or needed
    if args.train_model:
        train_model()
    
    # Run Streamlit app
    if not args.no_run:
        run_streamlit_app()
    
    logger.info("Application finished")

if __name__ == "__main__":
    main()