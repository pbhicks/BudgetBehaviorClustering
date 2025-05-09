#!/usr/bin/env python3
"""
Budget Behavior Clustering - Project Creator

This script creates all the necessary files and directories for the
Budget Behavior Clustering project from scratch.
"""

import os
import sys
import json
import argparse

def create_directory_structure():
    """Create the necessary directory structure."""
    # Define base and target structure
    base_dir = os.getcwd()
    
    # Backend directories
    backend_dir = os.path.join(base_dir, "backend")
    backend_data_dir = os.path.join(backend_dir, "data")
    backend_models_dir = os.path.join(backend_dir, "models")
    backend_logs_dir = os.path.join(backend_dir, "logs")
    
    # Create directories
    directories = [
        backend_dir, 
        backend_data_dir,
        backend_models_dir,
        backend_logs_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created")
    return (base_dir, backend_dir, backend_data_dir)

def create_file(file_path, content, file_type="file"):
    """Create a file with the specified content."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        print(f"ℹ️ {file_type.capitalize()} already exists: {os.path.relpath(file_path, os.getcwd())}")
        return False
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📝 Created {file_type}: {os.path.relpath(file_path, os.getcwd())}")
        return True
    except Exception as e:
        print(f"❌ Error creating {file_type}: {os.path.basename(file_path)} - {e}")
        return False

def create_readme_file(base_dir):
    """Create README.md file."""
    readme_md = """# 💸 Budget Behavior Clustering

A modular Streamlit application built to help users understand and optimize their financial behavior. Using unsupervised machine learning (KMeans clustering), this tool classifies users into financial personas based on their spending patterns—empowering better budgeting decisions through data.

Developed as part of a graduate finance capstone at the University of Arkansas, this app supports both direct budget entry and transaction data upload, with interactive visual feedback and tailored recommendations.

## 🧠 Overview

This project blends machine learning with personal finance UX:

* **KMeans Clustering** to identify user personas
* **Streamlit UI** with dual input modes
* **Plotly Visualizations** for insights
* **Sample Data** for instant testing
* **Modular Python Architecture** for extensibility

## ✨ Key Features

- **Budget Analysis**: Enter your income and expense data to receive a detailed analysis
- **Financial Persona Identification**: Discover your financial behavior pattern
- **Personalized Recommendations**: Get tailored recommendations
- **Interactive Visualizations**: Explore your budget data 
- **Financial Health Score**: Receive a score and rating

## 👤 Author

**Payton Hicks**  
MS Finance Candidate – University of Arkansas  
Cost Accountant | Builder of Tools | Advocate for Financial Clarity  
"""
    
    # Create the file
    create_file(os.path.join(base_dir, "README.md"), readme_md, "README")
    
    return True

def create_requirements_file(base_dir):
    """Create requirements.txt file."""
    requirements_txt = """# Budget Behavior Clustering - Requirements
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
plotly>=5.5.0
streamlit>=1.10.0
joblib>=1.1.0
"""
    
    # Create the file
    create_file(os.path.join(base_dir, "requirements.txt"), requirements_txt, "requirements")
    
    return True

def main():
    """Main entry point for the application."""
    # Create the directory structure
    base_dir, backend_dir, backend_data_dir = create_directory_structure()
    
    # Create requirements.txt
    requirements_created = create_requirements_file(base_dir)
    print(f"Requirements file created: {requirements_created}")
    
    # Create README.md
    readme_created = create_readme_file(base_dir)
    print(f"README file created: {readme_created}")
    
    # Print success message
    print("\n🎉 Budget Behavior Clustering project structure created successfully!")
    print("Next steps:")
    print("1. Create a virtual environment: python -m venv .venv")
    print("2. Activate the virtual environment: source .venv/bin/activate (or .venv\\Scripts\\activate on Windows)")
    print("3. Install dependencies: pip install -r requirements.txt")
    
if __name__ == "__main__":
    main()