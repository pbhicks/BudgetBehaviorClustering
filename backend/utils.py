"""
Budget Behavior Clustering - Utility Functions

This module contains utility functions for the Budget Behavior Clustering app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple

def setup_environment():
    """Setup the application environment."""
    # You could add environment variable setup or other configuration here
    os.makedirs("backend/data", exist_ok=True)
    os.makedirs("backend/models", exist_ok=True)
    os.makedirs("backend/logs", exist_ok=True)
    return True

def initialize_logging(level="INFO"):
    """Initialize logging for the application."""
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Configure logging
    log_dir = os.path.join("backend", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.info("Logging initialized")
    
    return logger

def tooltip(text, help_text):
    """Create an inline tooltip with help text."""
    return f"""
    <div class="tooltip">{text} <span>ℹ️</span>
        <span class="tooltiptext">{help_text}</span>
    </div>
    """

def get_download_link(file_content, file_name, link_text):
    """Generate a download link for a file."""
    b64 = base64.b64encode(file_content.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{file_name}" class="download-button">{link_text}</a>'

def process_transaction_file(uploaded_file):
    """Process an uploaded transaction file."""
    try:
        # Determine file type and read
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Basic validation
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Process date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                st.warning(f"Found {invalid_dates} rows with invalid dates. These will be excluded from analysis.")
                df = df.dropna(subset=['Date'])
        
        # Process amount column
        if 'Amount' in df.columns:
            # Convert amount to numeric, handling currency symbols and commas
            df['Amount'] = df['Amount'].astype(str)
            df['Amount'] = df['Amount'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
        
        # Check if there's data after filtering
        if df.empty:
            st.error("No valid data found in the file after processing.")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def apply_theme():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        /* Background and font */
        .main {
            background-color: #212529;  /* Dark background */
            color: #f8f9fa;  /* Light text */
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
        
        /* Make text light colored for visibility on dark backgrounds */
        p, label, span, div, li {
            color: #f8f9fa !important;
        }
        
        /* File uploader text */
        .uploadedFile, .uploadedFile span, .css-1aehpvj, .css-12oz5g7 {
            color: #f8f9fa !important;
        }
        
        /* Make file uploader area more visible */
        [data-testid="stFileUploader"] {
            background-color: #2c3034;  /* Slightly lighter than main background */
            border: 1px dashed #6c757d;
            padding: 10px;
            border-radius: 5px;
        }
        
        [data-testid="stFileUploader"] label, 
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p {
            color: #f8f9fa !important;
        }
        
        /* Inputs and sliders */
        .stSlider, .stNumberInput {
            color: #f8f9fa !important;
        }
        
        /* Header colors for dark theme */
        h1, h2, h3, h4 {
            color: #8bb9fe !important;  /* Light blue for headers */
        }
        
        /* Customize sidebar for dark theme */
        section[data-testid="stSidebar"] {
            background-color: #1a1d20;
            border-right: 1px solid #343a40;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #8bb9fe !important;
        }
        
        /* Rest of your CSS remains the same but adapting colors for dark theme... */
        
        /* Button styling for dark theme */
        .stButton button {
            background-color: #3b71ca;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-weight: 600;
        }
        
        .stButton button:hover {
            background-color: #2c5bb6;
        }
        
        /* Download button styling */
        .download-button {
            display: inline-block;
            padding: 8px 16px;
            background-color: #3b71ca;
            color: white !important;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 500;
            margin-top: 10px;
            text-align: center;
        }
        
        .download-button:hover {
            background-color: #2c5bb6;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)