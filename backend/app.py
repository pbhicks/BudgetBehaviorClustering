"""
Budget Behavior Clustering - Main App

This is the main entry point for the Streamlit app that provides
budget analysis, transaction categorization, and financial persona clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import List, Tuple, Dict, Optional

# Import from other modules
from utils import apply_theme, get_download_link
from cluster import (
    predict_cluster,
    get_persona_description,
    get_key_insights,
    get_budget_recommendations,
    calculate_financial_health_score
)
from visuals import (
    plot_comparison_chart,
    plot_radar_chart,
    plot_financial_health_gauge,
    plot_cluster_3d
)
from sample_data import generate_sample_data

# Set page configuration
st.set_page_config(
    page_title="Budget Behavior Clustering",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants for the app
PERSONA_COLORS = {
    "Cautious Saver": "#3B82F6",
    "Balanced Spender": "#10B981",
    "Aggressive Investor": "#F59E0B"
}

PERSONA_DESCRIPTIONS = {
    "Cautious Saver": "You prioritize saving and are careful with expenses. You tend to save more than average and spend less on discretionary items.",
    "Balanced Spender": "You maintain a healthy balance between spending and saving. Your budget allocations are well-proportioned across categories.",
    "Aggressive Investor": "You allocate more funds to investments and growth opportunities. You might spend more on quality items but maintain strategic financial goals."
}

FEATURE_NAMES = ["Income", "Housing", "Food", "Entertainment", "Savings Rate"]

# Apply custom CSS theme
apply_theme()

# Load Sample Data
@st.cache_data
def load_sample_data():
    """Load and return sample user budget data from CSV file."""
    try:
        data_path = os.path.join("backend", "data", "Sample_Users_Budget_Profile.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join("data", "Sample_Users_Budget_Profile.csv")
            
        if not os.path.exists(data_path):
            st.error("Sample data file not found. Generating sample data...")
            df = generate_sample_data(50)  # Generate 50 sample users
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            return df
        else:
            df = pd.read_csv(data_path)
            return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame(columns=FEATURE_NAMES)

# Normalize Data
def normalize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Normalize the dataset for clustering."""
    # Only keep numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    return pd.DataFrame(scaled, columns=numeric_df.columns), scaler

# Clustering
def cluster_data(normalized_df: pd.DataFrame, n_clusters: int = 3) -> KMeans:
    """Perform KMeans clustering on normalized data."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(normalized_df)
    return model

def get_cluster_averages(df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    """Calculate average values for each cluster."""
    # Only use numeric data
    features = df.select_dtypes(include=[np.number])
    features = features.copy()
    features['Cluster'] = model.labels_
    return features.groupby("Cluster").mean()

# UI Functions
def sidebar_inputs():
    """Render Streamlit sidebar for budget input."""
    with st.sidebar.form("budget_form"):
        st.subheader("💰 Monthly Income")
        income = st.number_input(
            "Enter your total monthly income ($)",
            min_value=0,
            max_value=100000,
            value=3500,
            step=100,
            help="Your total monthly income before taxes"
        )
        
        st.subheader("🏠 Essential Expenses")
        st.caption("Necessary living expenses you can't easily reduce")
        
        housing = st.number_input(
            "Housing/Rent ($)",
            min_value=0,
            max_value=50000,
            value=1200,
            step=50,
            help="Monthly housing costs including rent/mortgage, utilities"
        )
        
        food = st.number_input(
            "Food & Groceries ($)",
            min_value=0,
            max_value=5000,
            value=400,
            step=50,
            help="Monthly spending on groceries and dining"
        )
        
        st.subheader("🎭 Discretionary Spending")
        st.caption("Flexible expenses you have more control over")
        
        entertainment = st.number_input(
            "Entertainment ($)",
            min_value=0,
            max_value=5000,
            value=200,
            step=50,
            help="Monthly spending on entertainment, subscriptions, etc."
        )
        
        st.subheader("💹 Financial Goals")
        st.caption("What percentage of income would you like to save?")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            savings_rate = st.slider(
                "Savings Rate (%)",
                min_value=0,
                max_value=50,
                value=15,
                step=1,
                help="Percentage of income saved or invested monthly"
            )
        
        st.write("#### Ranges:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("📉 **Low**: 0-9%")
        with col2:
            st.caption("⚖️ **Average**: 10-15%")
        with col3:
            st.caption("📈 **High**: 16%+")
        
        submitted = st.form_submit_button("Analyze My Budget")
    
    if submitted:
        # Convert savings rate from percentage to decimal
        savings_decimal = savings_rate / 100.0
        
        # Return the user inputs as a list
        return [income, housing, food, entertainment, savings_decimal]
    
    return None

def display_transaction_upload():
    """Render transaction data upload interface."""
    st.sidebar.write("### 📂 Upload Transaction Data")
    st.sidebar.write("Upload your transaction history to get a more detailed analysis.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your transaction data for detailed analysis."
    )
    
    if uploaded_file is not None:
        st.sidebar.success("✅ File uploaded successfully!")
        return uploaded_file
    
    st.sidebar.write("### 📝 Need a template?")
    template_data = "Date,Category,Amount,Description\n2023-01-01,Housing,-1200.00,Rent\n2023-01-05,Food,-85.50,Groceries"
    st.sidebar.markdown(
        get_download_link(template_data, "transaction_template.csv", "📥 Download Template"),
        unsafe_allow_html=True
    )
    
    use_sample = st.sidebar.checkbox("Use sample data for demonstration")
    if use_sample:
        return "sample"
    
    return None

def display_persona_info(persona: str, description: str):
    """Display the user's financial persona information."""
    color = PERSONA_COLORS.get(persona, "#3B82F6")
    
    st.markdown(f"""
    <div style="
        background-color: {color}15;
        border-left: 5px solid {color};
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    ">
        <h3 style="color: {color}; margin-top: 0;">You are a {persona} 🧠</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def display_budget_summary(user_input: List[float]):
    """Display a summary of the user's budget."""
    income, housing, food, entertainment, savings_rate = user_input
    
    # Calculate metrics
    total_expenses = housing + food + entertainment
    savings_amount = income * savings_rate
    remaining = income - total_expenses - savings_amount
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Income", f"${income:.2f}")
    
    with col2:
        st.metric("Expenses", f"${total_expenses:.2f}")
    
    with col3:
        st.metric("Savings", f"${savings_amount:.2f}")
    
    with col4:
        st.metric(
            "Remaining", 
            f"${remaining:.2f}", 
            delta="positive" if remaining >= 0 else "negative"
        )

def display_budget_analysis(user_input: List[float], cluster_avg: List[float], persona: str):
    """Display comprehensive budget analysis."""
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Budget Comparison", 
        "📈 Budget Profile", 
        "🎯 Financial Health", 
        "🔍 Cluster Map"
    ])
    
    with tab1:
        st.plotly_chart(
            plot_comparison_chart(user_input, cluster_avg, FEATURE_NAMES),
            use_container_width=True
        )
    
    with tab2:
        st.plotly_chart(
            plot_radar_chart(user_input, cluster_avg, FEATURE_NAMES),
            use_container_width=True
        )
    
    with tab3:
        # Calculate financial health
        health_score, category = calculate_financial_health_score(user_input)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(
                plot_financial_health_gauge(health_score),
                use_container_width=True
            )
        
        with col2:
            st.write("### Financial Health Assessment")
            st.write(f"**Score:** {health_score:.1f}/100")
            st.write(f"**Category:** {category}")
            
            if category in ["Excellent", "Good"]:
                st.success("Your financial health is strong! Keep up the good work.")
            elif category == "Fair":
                st.info("Your financial health is reasonable, but there's room for improvement.")
            else:
                st.warning("Your financial health could use some attention. Consider the recommendations below.")
    
    with tab4:
        # Get sample data for 3D cluster visualization
        sample_df = st.session_state.get('sample_data')
        if sample_df is not None and len(sample_df) > 0:
            st.plotly_chart(
                plot_cluster_3d(
                    sample_df,
                    st.session_state.kmeans_model, 
                    user_input, 
                    st.session_state.cluster_idx
                ),
                use_container_width=True
            )
        else:
            st.info("Cluster map visualization requires sample data to be loaded.")

def display_recommendations(user_input: List[float], persona: str):
    """Display personalized recommendations based on the user's financial persona."""
    # Get insights
    insights = get_key_insights(user_input, persona)
    recommendations = get_budget_recommendations(user_input, persona)
    
    # Display in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 💡 Key Insights")
        for insight in insights:
            st.markdown(f"- {insight}")
    
    with col2:
        st.write("### 🎯 Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")

def display_welcome():
    """Display welcome and instruction message."""
    st.markdown("""
    # 💰 Budget Behavior Clustering
    
    This app analyzes your budget and classifies you into a financial persona based on your spending patterns.
    
    ### How it works:
    1. Enter your monthly budget details in the sidebar
    2. Submit to get your financial persona
    3. Explore your budget analysis and recommendations
    
    Get started by filling out the form in the sidebar! 👈
    """)

# Main Application
def main():
    """Main application function."""
    # Display welcome message if no data submitted yet
    if 'kmeans_model' not in st.session_state:
        display_welcome()
    
    # Get user input from sidebar
    user_input = sidebar_inputs()
    
    # Also check for transaction data upload
    transaction_data = display_transaction_upload()
    
    # Process user input
    if user_input or transaction_data:
        # Load sample data
        if 'sample_data' not in st.session_state:
            sample_data = load_sample_data()
            st.session_state.sample_data = sample_data
        
        # Use sample data if available
        if 'sample_data' in st.session_state and not st.session_state.sample_data.empty:
            # Normalize data
            if 'normalized_data' not in st.session_state or 'scaler' not in st.session_state:
                normalized_data, scaler = normalize_data(st.session_state.sample_data)
                st.session_state.normalized_data = normalized_data
                st.session_state.scaler = scaler
            
            # Create clustering model
            if 'kmeans_model' not in st.session_state:
                kmeans_model = cluster_data(st.session_state.normalized_data)
                st.session_state.kmeans_model = kmeans_model
                
                # Get cluster centers
                cluster_avgs = get_cluster_averages(st.session_state.sample_data, kmeans_model)
                st.session_state.cluster_avgs = cluster_avgs
            
            # Process transaction data if available
            if transaction_data == "sample":
                # Use sample data for demonstration
                st.info("Using sample transaction data for demonstration")
                # In a real app, we would load and process sample transactions here
                
            elif transaction_data is not None:
                # Process uploaded transaction data
                # In a real app, we would process the uploaded file here
                st.info("Transaction analysis will be available in the next version")
            
            # Process user input
            if user_input:
                # Predict cluster
                cluster_idx = predict_cluster(
                    user_input, 
                    st.session_state.kmeans_model, 
                    st.session_state.scaler
                )
                st.session_state.cluster_idx = cluster_idx
                
                # Get persona and description
                persona_mapping = {0: "Cautious Saver", 1: "Balanced Spender", 2: "Aggressive Investor"}
                persona = persona_mapping.get(cluster_idx, f"Cluster {cluster_idx}")
                description = get_persona_description(persona)
                
                # Get cluster average
                cluster_avg = st.session_state.cluster_avgs.loc[cluster_idx].values.tolist()
                
                # Display results
                st.title(f"Your Financial Profile: {persona}")
                
                # Display financial persona information
                display_persona_info(persona, description)
                
                # Display budget summary
                display_budget_summary(user_input)
                
                # Display budget analysis
                display_budget_analysis(user_input, cluster_avg, persona)
                
                # Display recommendations
                display_recommendations(user_input, persona)
        else:
            st.error("Cannot perform analysis without sample data. Please check data directory.")

if __name__ == "__main__":
    main()