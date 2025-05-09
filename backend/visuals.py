"""
Budget Behavior Clustering - Visualization Module

This module handles the visualization functions for the Budget Behavior Clustering app.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional

# Define colors for plotting
COLORS = {
    "primary": "#3B82F6",   # Blue
    "secondary": "#10B981", # Green
    "warning": "#F59E0B",   # Orange
    "danger": "#EF4444",    # Red
    "purple": "#8B5CF6",    # Purple
    "pink": "#EC4899",      # Pink
}

def plot_comparison_chart(user_input: List[float], cluster_avg: List[float], feature_names: List[str]):
    """Create a bar chart comparing user input to cluster average.
    
    Args:
        user_input: User's budget values
        cluster_avg: Average values for user's assigned cluster
        feature_names: Names of the features
        
    Returns:
        Plotly figure with comparison chart
    """
    # Create the data arrays
    categories = feature_names
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            name='Your Budget',
            x=categories,
            y=user_input,
            marker_color=COLORS["primary"],
            text=[f"${v:.2f}" if i < 4 else f"{v*100:.1f}%" for i, v in enumerate(user_input)],
            textposition='auto'
        ),
        go.Bar(
            name='Cluster Average',
            x=categories,
            y=cluster_avg,
            marker_color=COLORS["secondary"],
            text=[f"${v:.2f}" if i < 4 else f"{v*100:.1f}%" for i, v in enumerate(cluster_avg)],
            textposition='auto'
        )
    ])
    
    # Customize layout
    fig.update_layout(
        title={
            'text': 'Your Budget vs. Cluster Average',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        barmode='group',
        xaxis_title='Budget Category',
        yaxis_title='Amount ($)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=40),
        height=500,
        template='plotly_white'
    )
    
    # Add a horizontal line at 0
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(categories)-0.5,
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    return fig

def plot_radar_chart(user_input: List[float], cluster_avg: List[float], feature_names: List[str]):
    """Create a radar/spider chart comparing user input to cluster average.
    
    Args:
        user_input: User's budget values
        cluster_avg: Average values for user's assigned cluster
        feature_names: Names of the features
        
    Returns:
        Plotly figure with radar chart
    """
    # Normalize the data for better visualization
    max_values = np.maximum(user_input, cluster_avg)
    user_norm = [u / m if m > 0 else 0 for u, m in zip(user_input, max_values)]
    cluster_norm = [c / m if m > 0 else 0 for c, m in zip(cluster_avg, max_values)]
    
    # Add first point at the end to close the polygon
    theta = feature_names
    user_norm.append(user_norm[0])
    cluster_norm.append(cluster_norm[0])
    theta.append(feature_names[0])
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatterpolar(
        r=user_norm,
        theta=theta,
        fill='toself',
        name='Your Budget',
        line_color=COLORS["primary"],
        fillcolor=f'rgba{tuple(int(COLORS["primary"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=cluster_norm,
        theta=theta,
        fill='toself',
        name='Cluster Average',
        line_color=COLORS["secondary"],
        fillcolor=f'rgba{tuple(int(COLORS["secondary"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title={
            'text': 'Budget Profile Comparison',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_financial_health_gauge(score: float):
    """Create a gauge chart showing financial health score.
    
    Args:
        score: Financial health score (0-100)
        
    Returns:
        Plotly figure with gauge chart
    """
    # Determine color based on score
    if score >= 80:
        color = "#10B981"  # Green - Excellent
    elif score >= 70:
        color = "#3B82F6"  # Blue - Good
    elif score >= 60:
        color = "#F59E0B"  # Orange - Fair
    elif score >= 50:
        color = "#FB923C"  # Light Orange - Needs Improvement
    else:
        color = "#EF4444"  # Red - Requires Attention
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text="Financial Health Score", font=dict(size=24)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="darkblue"),
            bar=dict(color=color),
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, 50], color="rgba(239, 68, 68, 0.2)"),
                dict(range=[50, 60], color="rgba(251, 146, 60, 0.2)"),
                dict(range=[60, 70], color="rgba(245, 158, 11, 0.2)"),
                dict(range=[70, 80], color="rgba(59, 130, 246, 0.2)"),
                dict(range=[80, 100], color="rgba(16, 185, 129, 0.2)")
            ],
            threshold=dict(
                line=dict(color="red", width=4),
                thickness=0.75,
                value=score
            )
        )
    ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        template='plotly_white'
    )
    
    return fig

def plot_cluster_3d(df: pd.DataFrame, model: KMeans, user_input: List[float], user_cluster: int):
    """Create a 3D scatter plot of clusters using the first 3 principal components.
    
    Args:
        df: Original DataFrame with features
        model: Fitted KMeans model
        user_input: User's budget values
        user_cluster: Predicted cluster for user
        
    Returns:
        Plotly figure with 3D cluster visualization
    """
    # Prepare the data - keep only numeric columns
    features = df.select_dtypes(include=[np.number])
    
    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features)
    
    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2', 'PC3']
    )
    
    # Add cluster labels
    pca_df['Cluster'] = model.labels_
    
    # Map cluster to persona name
    persona_map = {0: "Cautious Saver", 1: "Balanced Spender", 2: "Aggressive Investor"}
    pca_df['Persona'] = pca_df['Cluster'].map(lambda x: persona_map.get(x, f"Cluster {x}"))
    
    # Transform user input
    user_pca = pca.transform(np.array(user_input).reshape(1, -1))
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Persona',
        color_discrete_map={
            "Cautious Saver": COLORS["primary"],
            "Balanced Spender": COLORS["secondary"],
            "Aggressive Investor": COLORS["warning"]
        },
        opacity=0.7,
        title="Budget Clusters in 3D Space"
    )
    
    # Add user point
    fig.add_scatter3d(
        x=[user_pca[0, 0]],
        y=[user_pca[0, 1]],
        z=[user_pca[0, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond'
        ),
        name='You'
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5
        ),
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3"
        )
    )
    
    return fig

def plot_spending_trend(df: pd.DataFrame):
    """Create a line chart showing spending trends over time.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Plotly figure with line chart
    """
    # Group by month and category, summing the amounts
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    monthly_spending = df[df['Amount'] < 0].groupby(['Month', 'Category'])['Amount'].sum().abs().reset_index()
    
    # Pivot the data
    pivot_df = monthly_spending.pivot(index='Month', columns='Category', values='Amount')
    
    # Plot the data
    fig = px.line(
        pivot_df, 
        x=pivot_df.index, 
        y=pivot_df.columns,
        title='Monthly Spending by Category',
        labels={'value': 'Amount ($)', 'x': 'Month'}
    )
    
    # Update layout
    fig.update_layout(
        legend_title_text='Category',
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_category_pie(df: pd.DataFrame):
    """Create a pie chart showing spending distribution by category.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Plotly figure with pie chart
    """
    # Group by category, summing the amounts (only expenses)
    category_spending = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs().reset_index()
    
    # Plot the data
    fig = px.pie(
        category_spending, 
        values='Amount', 
        names='Category',
        title='Spending Distribution by Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    
    # Update traces
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hole=0.4
    )
    
    return fig