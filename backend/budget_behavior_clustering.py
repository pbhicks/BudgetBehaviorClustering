"""Budget Behavior Clustering App

This Streamlit app collects user inputs for monthly budget data,
compares it against a sample dataset, and uses KMeans clustering
to classify the user into one of three financial behavior clusters.
Displays the user's predicted cluster and visualizes how their
budget compares to their cluster's average.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os
from typing import List, Tuple, Dict

# === Constants ===
CLUSTER_PERSONAS = {
    0: "Cautious Saver",
    1: "Balanced Spender", 
    2: "Aggressive Investor"
}

CLUSTER_DESCRIPTIONS = {
    0: "You prioritize saving and are careful with expenses. You tend to save more than average and spend less on discretionary items.",
    1: "You maintain a healthy balance between spending and saving. Your budget allocations are well-proportioned across categories.",
    2: "You allocate more funds to investments and growth opportunities. You might spend more on quality items but maintain strategic financial goals."
}

FEATURE_NAMES = ["Income", "Housing", "Food", "Entertainment", "Savings Rate"]

# === Data Loading Functions ===
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load and return sample user budget data from CSV file.
    
    Returns:
        pd.DataFrame: Sample budget data with features and labels
    """
    # Check if data directory exists
    if not os.path.exists("data"):
        st.error("Data directory not found. Please create a 'data' directory and add the sample_users.csv file.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv("data/Sample_Users_Budget_Profile.csv")

        return df
    except FileNotFoundError:
        st.error("Sample data file not found. Please ensure 'sample_users.csv' exists in the 'data' directory.")
        # Create empty DataFrame with correct columns
        return pd.DataFrame(columns=FEATURE_NAMES + ["label"])
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()


# === Clustering Functions ===
def normalize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Normalize the dataset for clustering.
    
    Args:
        df: DataFrame containing budget features
        
    Returns:
        Tuple of normalized DataFrame and the scaler object
    """
    # Extract features (drop the label column if it exists)
    features = df.drop(columns=["label"]) if "label" in df.columns else df
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    return scaled_df, scaler


def cluster_data(normalized_df: pd.DataFrame, n_clusters: int = 3) -> KMeans:
    """Perform KMeans clustering on normalized data.
    
    Args:
        normalized_df: Normalized DataFrame
        n_clusters: Number of clusters to form
        
    Returns:
        Fitted KMeans model
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    kmeans.fit(normalized_df)
    return kmeans


def predict_cluster(user_input: List[float], model: KMeans, scaler: StandardScaler) -> int:
    """Predict the cluster for a user's budget data.
    
    Args:
        user_input: List of user budget values
        model: Trained KMeans model
        scaler: Scaler used to normalize training data
        
    Returns:
        Predicted cluster index
    """
    # Convert user input to numpy array and reshape
    user_array = np.array(user_input).reshape(1, -1)
    
    # Normalize using the same scaler
    user_normalized = scaler.transform(user_array)
    
    # Predict cluster
    cluster = model.predict(user_normalized)[0]
    
    return cluster


def get_cluster_averages(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Calculate average values for each feature within each cluster.
    
    Args:
        df: Original DataFrame with features
        kmeans: Fitted KMeans model
        
    Returns:
        DataFrame with average values for each cluster
    """
    # Extract features
    features = df.drop(columns=["label"]) if "label" in df.columns else df
    
    # Add cluster predictions to original data
    features_with_clusters = features.copy()
    features_with_clusters['Cluster'] = kmeans.labels_
    
    # Calculate means for each cluster
    cluster_means = features_with_clusters.groupby('Cluster').mean()
    
    return cluster_means


# === Visualization Functions ===
def plot_comparison_chart(user_input: List[float], cluster_avg: List[float]) -> plt.Figure:
    """Create a bar chart comparing user input to cluster average.
    
    Args:
        user_input: User's budget values
        cluster_avg: Average values for user's assigned cluster
        
    Returns:
        Matplotlib figure with comparison chart
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the bar chart
    x = np.arange(len(FEATURE_NAMES))
    width = 0.35
    
    # Plot the bars
    ax.bar(x - width/2, user_input, width, label='Your Budget', color='#3E92CC')
    ax.bar(x + width/2, cluster_avg, width, label='Cluster Average', color='#43AA8B')
    
    # Customize the chart
    ax.set_ylabel('Amount', fontsize=12)
    ax.set_title('Your Budget vs. Cluster Average', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, fontsize=10, rotation=30, ha='right')
    ax.legend(fontsize=10)
    
    # Add value labels on top of each bar
    for i, v in enumerate(user_input):
        ax.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
    
    for i, v in enumerate(cluster_avg):
        ax.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
    
    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_radar_chart(user_input: List[float], cluster_avg: List[float]) -> plt.Figure:
    """Create a radar/spider chart comparing user input to cluster average.
    
    Args:
        user_input: User's budget values
        cluster_avg: Average values for user's assigned cluster
        
    Returns:
        Matplotlib figure with radar chart
    """
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(FEATURE_NAMES)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the user data
    user_values = user_input + user_input[:1]  # Close the loop
    ax.plot(angles, user_values, 'o-', linewidth=2, label='Your Budget')
    ax.fill(angles, user_values, alpha=0.1)
    
    # Add the cluster average data
    cluster_values = cluster_avg + cluster_avg[:1]  # Close the loop
    ax.plot(angles, cluster_values, 'o-', linewidth=2, label='Cluster Average')
    ax.fill(angles, cluster_values, alpha=0.1)
    
    # Set labels and customizations
    ax.set_thetagrids(np.degrees(angles[:-1]), FEATURE_NAMES)
    ax.set_title('Your Budget Profile vs. Cluster Average', fontsize=14, fontweight='bold')
    ax.grid(True)
    plt.legend(loc='upper right')
    
    return fig


def plot_clusters_3d(df: pd.DataFrame, kmeans: KMeans, user_input: List[float], user_cluster: int) -> plt.Figure:
    """Create a 3D scatter plot of clusters using the first 3 principal components.
    
    Args:
        df: Original DataFrame with features
        kmeans: Fitted KMeans model
        user_input: User's budget values
        user_cluster: Predicted cluster for user
        
    Returns:
        Matplotlib figure with 3D cluster visualization
    """
    from sklearn.decomposition import PCA
    
    # Prepare the data
    features = df.drop(columns=["label"]) if "label" in df.columns else df
    
    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features)
    
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cluster
    for cluster in range(kmeans.n_clusters):
        # Get points in this cluster
        mask = kmeans.labels_ == cluster
        cluster_points = principal_components[mask]
        
        # Plot cluster points
        ax.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            cluster_points[:, 2],
            label=f'Cluster {cluster}: {CLUSTER_PERSONAS[cluster]}',
            alpha=0.7
        )
    
    # Plot user point
    user_transformed = pca.transform(np.array(user_input).reshape(1, -1))
    ax.scatter(
        user_transformed[:, 0],
        user_transformed[:, 1],
        user_transformed[:, 2],
        color='red',
        s=100,
        label='You',
        marker='X'
    )
    
    # Set labels and title
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('Budget Clusters in 3D Space', fontsize=14, fontweight='bold')
    
    # Add a legend
    plt.legend()
    
    return fig


# === UI Elements ===
def apply_theme() -> None:
    """Apply custom CSS theme to Streamlit."""
    css = """
    <style>
        /* Background and font */
        .main { background-color: #F8F9FA; color: #212529; font-family: 'Segoe UI', sans-serif; }
        
        /* Header colors */
        h1, h2, h3 { color: #0B2447; }
        
        /* Custom container for metrics */
        .metric-container {
            background-color: #FFFFFF;
            border-radius: 7px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        /* Sidebar tweaks */
        section[data-testid="stSidebar"] { background-color: #F1F3F5; }
        
        /* Buttons */
        .stButton>button { 
            background-color: #0B2447; 
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 15px;
        }
        .stButton>button:hover { 
            background-color: #19376D; 
            border: none;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar() -> List[float]:
    """Render Streamlit sidebar form for budget input.
    
    Returns:
        List of user input values
    """
    st.sidebar.header("Enter Your Monthly Budget")
    
    with st.sidebar.form("budget_form"):
        # Monthly income
        income = st.number_input(
            "Monthly Income ($)",
            min_value=0,
            max_value=50000,
            value=3000,
            step=100,
            help="Your total monthly income before taxes"
        )
        
        # Housing expenses
        housing = st.number_input(
            "Housing/Rent ($)",
            min_value=0,
            max_value=20000,
            value=1000,
            step=50,
            help="Monthly housing costs including rent/mortgage, utilities"
        )
        
        # Food expenses
        food = st.number_input(
            "Food & Groceries ($)",
            min_value=0,
            max_value=5000,
            value=500,
            step=50,
            help="Monthly spending on groceries and dining"
        )
        
        # Entertainment expenses
        entertainment = st.number_input(
            "Entertainment ($)",
            min_value=0,
            max_value=5000,
            value=200,
            step=50,
            help="Monthly spending on entertainment, subscriptions, etc."
        )
        
        # Savings rate
        savings_rate = st.slider(
            "Savings Rate (%)",
            min_value=0,
            max_value=100,
            value=15,
            step=1,
            help="Percentage of income saved or invested monthly"
        )
        
        # Submit button
        submitted = st.form_submit_button("Analyze My Budget")
    
    # Process form submission
    if submitted:
        # Convert savings rate from percentage to decimal
        savings_decimal = savings_rate / 100.0
        
        # Return the user inputs as a list
        return [income, housing, food, entertainment, savings_decimal]
    
    return None


def display_cluster_results(user_input: List[float], cluster_idx: int, cluster_centers: pd.DataFrame) -> None:
    """Display clustering results and visualizations.
    
    Args:
        user_input: User's budget values
        cluster_idx: Predicted cluster index
        cluster_centers: DataFrame of cluster centers/averages
    """
    # Get the cluster persona and description
    persona = CLUSTER_PERSONAS[cluster_idx]
    description = CLUSTER_DESCRIPTIONS[cluster_idx]
    
    # Get the average values for the user's cluster
    cluster_avg = cluster_centers.loc[cluster_idx].values
    
    # Display the cluster results
    st.markdown(f"## 🎯 Your Budget Profile: **{persona}**")
    st.markdown(f"*{description}*")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Comparison Chart", "Radar Profile", "Cluster Map"])
    
    with tab1:
        # Bar chart comparison
        st.markdown("### Your Budget vs. Cluster Average")
        fig1 = plot_comparison_chart(user_input, cluster_avg)
        st.pyplot(fig1)
    
    with tab2:
        # Radar chart
        st.markdown("### Budget Profile Comparison")
        fig2 = plot_radar_chart(user_input, cluster_avg)
        st.pyplot(fig2)
    
    with tab3:
        # 3D cluster visualization
        st.markdown("### Your Position in Budget Clusters")
        # This will be populated if sample data is loaded
        if hasattr(st.session_state, 'sample_data') and not st.session_state.sample_data.empty:
            fig3 = plot_clusters_3d(
                st.session_state.sample_data, 
                st.session_state.kmeans_model, 
                user_input, 
                cluster_idx
            )
            st.pyplot(fig3)
        else:
            st.info("Cluster map visualization requires sample data to be loaded.")


def display_budget_insights(user_input: List[float], cluster_avg: List[float]) -> None:
    """Display budget insights and recommendations.
    
    Args:
        user_input: User's budget values
        cluster_avg: Average values for user's assigned cluster
    """
    # Calculate some metrics
    housing_percent = (user_input[1] / user_input[0]) * 100
    food_percent = (user_input[2] / user_input[0]) * 100
    entertainment_percent = (user_input[3] / user_input[0]) * 100
    savings_percent = user_input[4] * 100
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Housing", f"{housing_percent:.1f}%", 
                 f"{housing_percent - (cluster_avg[1]/cluster_avg[0])*100:.1f}%")
    
    with col2:
        st.metric("Food", f"{food_percent:.1f}%", 
                 f"{food_percent - (cluster_avg[2]/cluster_avg[0])*100:.1f}%")
    
    with col3:
        st.metric("Entertainment", f"{entertainment_percent:.1f}%", 
                 f"{entertainment_percent - (cluster_avg[3]/cluster_avg[0])*100:.1f}%")
    
    with col4:
        st.metric("Savings", f"{savings_percent:.1f}%", 
                 f"{savings_percent - cluster_avg[4]*100:.1f}%")
    
    # Generate insights
    st.markdown("### 💡 Budget Insights")
    
    # Housing insights
    if housing_percent > 35:
        st.warning("Your housing costs are above the recommended 30-35% of income. Consider ways to reduce housing expenses.")
    elif housing_percent < (cluster_avg[1]/cluster_avg[0])*100 - 10:
        st.success("Your housing costs are well below average for your cluster. Great job finding affordable housing!")
    
    # Savings insights
    if savings_percent < 10:
        st.warning("Your savings rate is below 10%. Consider increasing your savings for better financial security.")
    elif savings_percent > 20:
        st.success("Your savings rate exceeds 20%, which is excellent for building long-term wealth!")
    
    # Entertainment insights
    if entertainment_percent > 15:
        st.warning("Your entertainment spending is relatively high. Consider setting a budget for discretionary expenses.")
    
    # Overall budget balance
    remaining = user_input[0] - user_input[1] - user_input[2] - user_input[3] - (user_input[0] * user_input[4])
    if remaining < 0:
        st.error("Your budget shows a deficit. Your expenses and savings exceed your income.")
    else:
        st.info(f"You have ${remaining:.2f} unallocated in your budget that could be directed toward additional savings or debt reduction.")


# === Main Function ===
def main() -> None:
    """Main entry point: configure page, apply theme and render UI."""
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Budget Behavior Clustering",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom theme
    apply_theme()
    
    # Page title and introduction
    st.title("💰 Budget Behavior Clustering")
    st.markdown("""
    This app analyzes your monthly budget and compares it to others using machine learning.
    Enter your budget details in the sidebar to see which financial persona matches your spending habits.
    """)
    
    # Initialize session state for storing data and models
    if 'sample_data' not in st.session_state:
        # Load sample data
        sample_data = load_sample_data()
        st.session_state.sample_data = sample_data
        
        # If sample data was loaded successfully, train the clustering model
        if not sample_data.empty:
            # Normalize the data
            normalized_data, scaler = normalize_data(sample_data)
            st.session_state.scaler = scaler
            
            # Perform clustering
            kmeans = cluster_data(normalized_data)
            st.session_state.kmeans_model = kmeans
            
            # Get cluster averages
            cluster_avgs = get_cluster_averages(sample_data, kmeans)
            st.session_state.cluster_averages = cluster_avgs
            
            st.success("Sample data loaded and clustering model trained successfully!")
        else:
            st.warning("No sample data available. Please add sample_users.csv to the data directory.")
    
    # Render sidebar and get user inputs
    user_input = render_sidebar()
    
    # Process user input
    if user_input:
        # Check if we have sample data and a trained model
        if hasattr(st.session_state, 'sample_data') and not st.session_state.sample_data.empty:
            # Predict user's cluster
            cluster_idx = predict_cluster(
                user_input,
                st.session_state.kmeans_model,
                st.session_state.scaler
            )
            
            # Display results
            display_cluster_results(
                user_input,
                cluster_idx,
                st.session_state.cluster_averages
            )
            
            # Display budget insights
            st.markdown("---")
            display_budget_insights(
                user_input, 
                st.session_state.cluster_averages.loc[cluster_idx].values
            )
        else:
            st.error("Cannot perform clustering without sample data. Please ensure sample_users.csv is available.")
    else:
        # Display example/placeholder content when no user input is provided
        st.info("👈 Enter your budget details in the sidebar to get started!")
        
        # Show example visualization or explanation
        st.markdown("""
        ### How it works
        
        1. **Enter your budget data** in the sidebar form
        2. **Submit for analysis** to see your financial persona
        3. **View visualizations** comparing your budget to similar profiles
        4. **Get personalized insights** based on your spending patterns
        
        The app uses K-means clustering, a machine learning algorithm, to group similar budget profiles
        and identify patterns in financial behavior.
        """)


# Run the application
if __name__ == "__main__":
    main()