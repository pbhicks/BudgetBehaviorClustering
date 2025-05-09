"""
Budget Behavior Clustering - Clustering Module

This module contains the clustering logic for the Budget Behavior Clustering app.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

def train_clustering_model(df, model_path=None):
    """Train a KMeans clustering model and save it.
    
    Args:
        df (DataFrame): Data to train on
        model_path (Path, optional): Path to save model. Defaults to None.
    
    Returns:
        tuple: Trained model and scaled data
    """
    # Select features for clustering
    features = df.columns.tolist()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Train KMeans with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    
    # Save model if path provided
    if model_path:
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and scaler
        joblib.dump((kmeans, scaler), model_path)
    
    return kmeans, scaler

def load_model(model_path):
    """Load a saved clustering model and scaler.
    
    Args:
        model_path (str): Path to saved model
    
    Returns:
        tuple: Loaded model and scaler
    """
    if os.path.exists(model_path):
        kmeans, scaler = joblib.load(model_path)
        return kmeans, scaler
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def perform_clustering(df, model_path=None):
    """Perform clustering on the data.
    
    Args:
        df (DataFrame): Data to cluster
        model_path (str, optional): Path to saved model. Defaults to None.
        
    Returns:
        tuple: DataFrame with cluster assignments and model
    """
    # Check if model exists
    if model_path and os.path.exists(model_path):
        kmeans, scaler = load_model(model_path)
        scaled_data = scaler.transform(df)
    else:
        # Train new model
        kmeans, scaler = train_clustering_model(df, model_path)
        scaled_data = scaler.transform(df)
    
    # Assign clusters
    df_copy = df.copy()
    df_copy['Cluster'] = kmeans.predict(scaled_data)
    
    return df_copy, kmeans

def get_user_cluster(user_df, model):
    """Get the cluster assignment for a user.
    
    Args:
        user_df (DataFrame): User data
        model (KMeans): Trained KMeans model
        
    Returns:
        int: Cluster assignment
    """
    # Assign cluster
    user_cluster = model.predict(user_df)[0]
    return user_cluster

def get_persona_description(cluster_id):
    """Get the persona name and description for a cluster.
    
    Args:
        cluster_id (int): Cluster ID
        
    Returns:
        tuple: Persona name and description
    """
    personas = {
        0: {
            "name": "Cautious Saver",
            "description": "You are a careful planner who prioritizes saving and financial security. "
                          "You tend to be conservative with discretionary spending and focus on "
                          "building a strong savings buffer. Your approach helps you prepare for "
                          "emergencies and long-term goals, though you might benefit from "
                          "optimizing your investment strategy for better returns."
        },
        1: {
            "name": "Balanced Spender",
            "description": "You maintain a healthy balance between enjoying life today and "
                          "saving for tomorrow. Your spending habits are moderate, and you "
                          "allocate funds across needs and wants with consideration. "
                          "This balanced approach provides both financial stability and "
                          "quality of life, though you might benefit from increasing your "
                          "savings rate slightly for future goals."
        },
        2: {
            "name": "Aggressive Investor",
            "description": "You prioritize growth and are comfortable with higher-risk financial "
                          "strategies. Your spending on lifestyle and discretionary categories is "
                          "higher than average, but so is your income. Your approach can lead to "
                          "significant wealth building if managed well, though you might benefit "
                          "from increasing your emergency reserves for unexpected expenses."
        }
    }
    
    # Default in case of unexpected cluster
    default_persona = {
        "name": "Financial Persona",
        "description": "Your financial behavior shows a unique pattern that combines elements of "
                      "different financial approaches. Based on your spending and saving habits, "
                      "we've identified areas where you can optimize your budget to better "
                      "achieve your financial goals."
    }
    
    persona = personas.get(cluster_id, default_persona)
    return persona["name"], persona["description"]
def predict_cluster(user_data, model, scaler):
    """Predict the cluster for new user data.
    
    Args:
        user_data (array-like): User data (can be 1D array, list, or dict)
        model (KMeans): Trained KMeans model
        scaler (StandardScaler): Fitted scaler
        
    Returns:
        int: Predicted cluster
    """
    # Convert input to appropriate format
    if isinstance(user_data, dict):
        # If user_data is a dictionary, convert to array
        # Expected order: features that model was trained on
        user_array = np.array(list(user_data.values()))
    elif isinstance(user_data, (list, np.ndarray)):
        # If already an array-like object, use as is
        user_array = np.array(user_data)
    else:
        raise ValueError("user_data must be a dictionary, list, or numpy array")
    
    # Reshape to 2D array if it's 1D
    if user_array.ndim == 1:
        user_array = user_array.reshape(1, -1)
    
    # Scale the data
    scaled_data = scaler.transform(user_array)
    
    # Predict cluster
    cluster = model.predict(scaled_data)[0]
    
    return cluster
def get_key_insights(user_data, cluster_avg):
    """Generate key insights by comparing user data with cluster average.
    
    Args:
        user_data (dict): User budget data
        cluster_avg (Series): Cluster average data
        
    Returns:
        list: List of insights
    """
    insights = []
    
    # Compare key budget categories
    categories = {
        "Housing": "housing costs",
        "Food": "food expenses",
        "Entertainment": "entertainment spending",
        "Savings_Rate": "savings rate"
    }
    
    for category, label in categories.items():
        cat_key = category
        cluster_cat = category
        
        # Handle different naming conventions
        if category == "Savings_Rate" and "Savings Rate" in cluster_avg:
            cluster_cat = "Savings Rate"
        elif category == "Savings Rate" and "Savings_Rate" in user_data:
            cat_key = "Savings_Rate"
        
        if cat_key in user_data and cluster_cat in cluster_avg:
            user_val = user_data[cat_key]
            cluster_val = cluster_avg[cluster_cat]
            
            # For savings rate, convert to percentage
            if category in ["Savings_Rate", "Savings Rate"]:
                diff = (user_val - cluster_val) * 100  # Convert to percentage points
                if diff > 1:
                    insights.append(f"✅ Your {label} is {abs(diff):.1f}% higher than average for your persona.")
                elif diff < -1:
                    insights.append(f"⚠️ Your {label} is {abs(diff):.1f}% lower than average for your persona.")
                else:
                    insights.append(f"Your {label} is similar to the average for your persona.")
            else:
                # For expenses, calculate percentage difference
                pct_diff = ((user_val - cluster_val) / cluster_val) * 100
                if pct_diff < -5:
                    insights.append(f"✅ Your {label} are {abs(pct_diff):.1f}% lower than average for your persona.")
                elif pct_diff > 5:
                    insights.append(f"⚠️ Your {label} are {pct_diff:.1f}% higher than average for your persona.")
                else:
                    insights.append(f"Your {label} are similar to the average for your persona.")
    
    # Add income insight if available
    if "Income" in user_data and "Income" in cluster_avg:
        income_diff = ((user_data["Income"] - cluster_avg["Income"]) / cluster_avg["Income"]) * 100
        if income_diff > 10:
            insights.append(f"✅ Your income is {income_diff:.1f}% higher than average for your persona.")
        elif income_diff < -10:
            insights.append(f"⚠️ Your income is {abs(income_diff):.1f}% lower than average for your persona.")
        else:
            insights.append(f"Your income is similar to the average for your persona.")
    
    return insights
def get_budget_recommendations(user_data, cluster_id):
    """Generate personalized budget recommendations based on the user's data and cluster.
    
    Args:
        user_data (dict): User budget data
        cluster_id (int): User's cluster ID
        
    Returns:
        list: List of personalized recommendations
    """
    recommendations = []
    
    # Get persona type
    persona_name, _ = get_persona_description(cluster_id)
    
    # Calculate key metrics
    if "Income" in user_data:
        income = user_data["Income"]
        housing_ratio = user_data.get("Housing", 0) / income if income > 0 else 0
        food_ratio = user_data.get("Food", 0) / income if income > 0 else 0
        entertainment_ratio = user_data.get("Entertainment", 0) / income if income > 0 else 0
        savings_rate = user_data.get("Savings_Rate", 0)
        
        # Common recommendations based on financial rules of thumb
        if housing_ratio > 0.33:
            recommendations.append(
                "🏠 Your housing costs exceed the recommended 33% of income. Consider ways to "
                "reduce these costs or increase your income to improve financial stability."
            )
        
        if savings_rate < 0.10:
            recommendations.append(
                "💰 Aim to increase your savings rate to at least 10% of income. Even small, "
                "consistent increases can significantly build your emergency fund and long-term savings."
            )
        
        # Persona-specific recommendations
        if persona_name == "Cautious Saver":
            recommendations.append(
                "📈 While you're doing well with saving, consider diversifying your savings into "
                "investments that can provide better returns while maintaining relatively low risk, "
                "such as index funds or high-yield savings accounts."
            )
            
            if entertainment_ratio < 0.03:
                recommendations.append(
                    "🎭 It's important to balance saving with enjoying life. Consider allocating a small "
                    "budget for personal enjoyment to prevent 'savings fatigue' and maintain motivation."
                )
                
        elif persona_name == "Balanced Spender":
            recommendations.append(
                "⚖️ Your balanced approach is effective. Consider automating your savings and bill payments "
                "to maintain consistency, and review your budget quarterly to ensure it aligns with your goals."
            )
            
            if food_ratio > 0.15:
                recommendations.append(
                    "🍲 Your food expenses are somewhat high. Meal planning, grocery optimization, or "
                    "reducing dining out could free up funds for other priorities without sacrificing quality."
                )
                
        elif persona_name == "Aggressive Investor":
            recommendations.append(
                "🛡️ With your growth-oriented approach, ensure you maintain a sufficient emergency fund "
                "(3-6 months of expenses) as a safety net for your more aggressive financial strategies."
            )
            
            if entertainment_ratio > 0.10:
                recommendations.append(
                    "⚠️ Your discretionary spending is relatively high. Track this category closely to "
                    "ensure it's providing value proportional to the cost and doesn't impede long-term goals."
                )
    
    # General recommendations if we don't have enough specific ones
    general_recommendations = [
        "📊 Consider using a dedicated budgeting app or spreadsheet to track expenses more accurately.",
        "🔄 Review subscriptions and recurring expenses quarterly to eliminate those you no longer use or value.",
        "📱 Look into automated savings tools that can help you save without having to think about it.",
        "📝 Set specific, measurable financial goals for the short-term (1 year), medium-term (5 years), and long-term.",
        "💳 Prioritize high-interest debt reduction as part of your financial strategy.",
        "🏦 Consider tax-advantaged accounts like 401(k) or IRA to optimize your long-term savings strategy.",
        "📚 Invest time in financial education through books, courses, or podcasts to improve your money management skills."
    ]
    
    # Add general recommendations if needed
    while len(recommendations) < 3 and general_recommendations:
        recommendations.append(general_recommendations.pop(0))
    
    return recommendations
def calculate_financial_health_score(user_data):
    """Calculate a financial health score for the user.
    
    Args:
        user_data (dict or list): User budget data
        
    Returns:
        tuple: Score (0-100) and rating (Poor to Excellent)
    """
    # Initialize score
    score = 50  # Start at neutral
    
    # Handle different input types
    if isinstance(user_data, list):
        # If user_data is a list, assume it's in the order: [Income, Housing, Food, Entertainment, Savings_Rate]
        # If the list is shorter, fill with zeros
        padded_list = user_data + [0] * (5 - len(user_data)) if len(user_data) < 5 else user_data
        income = padded_list[0]
        housing = padded_list[1]
        food = padded_list[2]
        entertainment = padded_list[3]
        savings_rate = padded_list[4]
    else:
        # Assume user_data is a dictionary
        income = user_data.get("Income", 0)
        housing = user_data.get("Housing", 0)
        food = user_data.get("Food", 0)
        entertainment = user_data.get("Entertainment", 0)
        savings_rate = user_data.get("Savings_Rate", 0)
    
    # Calculate total expenses
    expenses = housing + food + entertainment
    
    # Calculate metrics
    housing_ratio = housing / income if income > 0 else 1
    debt_to_income = 0  # Placeholder, would use actual debt data if available
    
    # Adjust score based on metrics
    # Savings rate (weight: 30%)
    if savings_rate >= 0.20:  # >20% savings rate
        score += 20
    elif savings_rate >= 0.10:  # >10% savings rate
        score += 15
    elif savings_rate >= 0.05:  # >5% savings rate
        score += 10
    elif savings_rate > 0:  # >0% savings rate
        score += 5
    else:
        score -= 10
    
    # Housing ratio (weight: 25%)
    if housing_ratio <= 0.25:  # <25% of income on housing
        score += 15
    elif housing_ratio <= 0.33:  # <33% of income on housing
        score += 10
    elif housing_ratio <= 0.40:  # <40% of income on housing
        score += 0
    else:
        score -= 15
    
    # Expense ratio (weight: 25%)
    expense_ratio = expenses / income if income > 0 else 1
    if expense_ratio <= 0.50:  # <50% of income on expenses
        score += 15
    elif expense_ratio <= 0.70:  # <70% of income on expenses
        score += 10
    elif expense_ratio <= 0.85:  # <85% of income on expenses
        score += 0
    else:
        score -= 15
    
    # Cap score between 0 and 100
    score = max(0, min(100, score))
    
    # Determine rating
    if score >= 90:
        rating = "Excellent"
    elif score >= 75:
        rating = "Good"
    elif score >= 60:
        rating = "Fair"
    elif score >= 40:
        rating = "Needs Improvement"
    else:
        rating = "Poor"
    
    return round(score), rating