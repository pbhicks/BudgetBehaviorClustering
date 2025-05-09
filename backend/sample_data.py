"""
Budget Behavior Clustering - Sample Data Generator

This module generates sample data for the Budget Behavior Clustering app
when real user data is not available.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import calendar
from typing import List, Optional, Dict, Tuple

def generate_sample_data(num_samples: int = 50) -> pd.DataFrame:
    """Generate sample budget data for clustering.
    
    Args:
        num_samples: Number of sample users to generate
        
    Returns:
        DataFrame with sample budget data
    """
    # Define income ranges for different personas
    income_ranges = {
        "Cautious Saver": (2500, 4500),
        "Balanced Spender": (3000, 6000),
        "Aggressive Investor": (4000, 8000)
    }
    
    # Initialize data list
    data = []
    
    # Generate data for each persona
    for persona, (min_income, max_income) in income_ranges.items():
        # Generate approximately equal number of samples for each persona
        n_samples = num_samples // 3
        
        for _ in range(n_samples):
            # Generate income
            income = np.random.uniform(min_income, max_income)
            
            # Generate expense percentages based on persona
            if persona == "Cautious Saver":
                # Cautious savers spend less on discretionary items
                housing_pct = np.random.uniform(0.25, 0.35)
                food_pct = np.random.uniform(0.08, 0.12)
                entertainment_pct = np.random.uniform(0.02, 0.05)
                savings_rate = np.random.uniform(0.2, 0.3)
            
            elif persona == "Balanced Spender":
                # Balanced spenders have moderate allocations
                housing_pct = np.random.uniform(0.3, 0.4)
                food_pct = np.random.uniform(0.1, 0.15)
                entertainment_pct = np.random.uniform(0.05, 0.1)
                savings_rate = np.random.uniform(0.1, 0.2)
            
            else:  # Aggressive Investor
                # Aggressive investors spend more on discretionary items
                housing_pct = np.random.uniform(0.35, 0.45)
                food_pct = np.random.uniform(0.12, 0.18)
                entertainment_pct = np.random.uniform(0.08, 0.15)
                savings_rate = np.random.uniform(0.05, 0.15)
            
            # Calculate actual amounts
            housing = income * housing_pct
            food = income * food_pct
            entertainment = income * entertainment_pct
            
            # Create user record
            user = {
                "Income": round(income, 2),
                "Housing": round(housing, 2),
                "Food": round(food, 2),
                "Entertainment": round(entertainment, 2),
                "Savings Rate": round(savings_rate, 2)
            }
            
            data.append(user)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def generate_sample_transactions(num_users: int = 5, start_date: str = '2023-01-01', end_date: str = '2023-03-31') -> pd.DataFrame:
    """Generate sample transaction data for demonstration.
    
    Args:
        num_users: Number of users to generate transactions for
        start_date: Start date for transaction period (YYYY-MM-DD)
        end_date: End date for transaction period (YYYY-MM-DD)
        
    Returns:
        DataFrame with sample transaction data
    """
    # Define categories and their typical spending ranges
    categories = {
        'Housing': (-1500, -800),
        'Food': (-300, -100),
        'Transportation': (-150, -50),
        'Entertainment': (-200, -20),
        'Shopping': (-300, -50),
        'Income': (2500, 5000)
    }
    
    # Create descriptions for each category
    descriptions = {
        'Housing': ['Rent', 'Mortgage', 'Utilities', 'Internet', 'Electricity'],
        'Food': ['Grocery Store', 'Restaurant', 'Takeout', 'Coffee Shop', 'Fast Food'],
        'Transportation': ['Gas', 'Car Payment', 'Public Transit', 'Uber/Lyft', 'Parking'],
        'Entertainment': ['Movies', 'Streaming Service', 'Concert', 'Video Games', 'Sports Event'],
        'Shopping': ['Clothing', 'Electronics', 'Home Goods', 'Online Shopping', 'Department Store'],
        'Income': ['Salary', 'Paycheck', 'Direct Deposit', 'Freelance Payment', 'Contract Work']
    }
    
    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Generate data 
    data = []
    
    for user_id in range(1, num_users + 1):
        # Start with a different balance for each user
        balance = np.random.uniform(1000, 3000)
        
        # Iterate through each month in the date range
        current_date = start_date
        while current_date <= end_date:
            # Get month and number of days in month
            month = current_date.month
            year = current_date.year
            days_in_month = calendar.monthrange(year, month)[1]
            
            # Generate 15-20 transactions per month
            num_transactions = np.random.randint(15, 21)
            transaction_days = np.random.choice(range(1, days_in_month + 1), num_transactions, replace=False)
            transaction_days.sort()
            
            # Ensure income is received on typical days (1st and/or 15th)
            income_days = []
            if 1 in transaction_days or np.random.random() < 0.8:  # 80% chance of income on 1st
                income_days.append(1)
            if 15 in transaction_days or np.random.random() < 0.7:  # 70% chance of income on 15th
                income_days.append(15)
                
            for day in transaction_days:
                # Set transaction date
                txn_date = datetime(year, month, day)
                
                # Determine category
                if day in income_days:
                    category = 'Income'
                else:
                    # Exclude income from regular transactions
                    non_income_categories = [c for c in categories.keys() if c != 'Income']
                    category = np.random.choice(non_income_categories)
                
                # Generate amount based on category
                min_val, max_val = categories[category]
                amount = float(np.random.randint(min_val * 100, max_val * 100) / 100)
                
                # Update balance
                balance += amount
                
                # Select a description based on category
                description = np.random.choice(descriptions[category])
                
                # Create transaction record
                transaction = {
                    'User_ID': f'user_{user_id}',
                    'Date': txn_date,
                    'Description': description,
                    'Category': category,
                    'Amount': round(amount, 2),
                    'Balance': round(balance, 2)
                }
                
                data.append(transaction)
            
            # Move to the next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by User_ID and Date
    df = df.sort_values(['User_ID', 'Date']).reset_index(drop=True)
    
    return df

def calculate_budget_metrics(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate budgeting metrics from transaction data.
    
    Args:
        transactions_df: DataFrame containing transaction data
        
    Returns:
        DataFrame with budget metrics for each user
    """
    # Initialize list to store user metrics
    user_metrics = []
    
    # Get unique user IDs
    user_ids = transactions_df['User_ID'].unique()
    
    for user_id in user_ids:
        # Filter transactions for this user
        user_txns = transactions_df[transactions_df['User_ID'] == user_id]
        
        # Calculate total income
        income = user_txns[user_txns['Category'] == 'Income']['Amount'].sum()
        
        # Calculate spending by category (amounts are negative)
        housing = abs(user_txns[user_txns['Category'] == 'Housing']['Amount'].sum())
        food = abs(user_txns[user_txns['Category'] == 'Food']['Amount'].sum())
        entertainment = abs(user_txns[user_txns['Category'] == 'Entertainment']['Amount'].sum())
        transportation = abs(user_txns[user_txns['Category'] == 'Transportation']['Amount'].sum())
        shopping = abs(user_txns[user_txns['Category'] == 'Shopping']['Amount'].sum())
        
        # Calculate total expenses
        total_expenses = housing + food + entertainment + transportation + shopping
        
        # Calculate savings (income - expenses)
        savings = income - total_expenses
        
        # Calculate savings rate
        savings_rate = savings / income if income > 0 else 0
        
        # Create user metrics record
        metrics = {
            'User_ID': user_id,
            'Income': round(income, 2),
            'Housing': round(housing, 2),
            'Food': round(food, 2),
            'Entertainment': round(entertainment, 2),
            'Transportation': round(transportation, 2),
            'Shopping': round(shopping, 2),
            'Total_Expenses': round(total_expenses, 2),
            'Savings': round(savings, 2),
            'Savings_Rate': round(savings_rate, 2)
        }
        
        user_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(user_metrics)
    
    return metrics_df

def save_sample_data(output_dir: str = 'sample_data'):
    """Generate and save sample data files.
    
    Args:
        output_dir: Directory to save sample data files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate clustering data
    clustering_df = generate_sample_data(num_samples=100)
    clustering_df.to_csv(os.path.join(output_dir, 'budget_clustering_data.csv'), index=False)
    
    # Generate transaction data
    transactions_df = generate_sample_transactions(num_users=10)
    transactions_df.to_csv(os.path.join(output_dir, 'sample_transactions.csv'), index=False)
    
    # Calculate and save budget metrics
    metrics_df = calculate_budget_metrics(transactions_df)
    metrics_df.to_csv(os.path.join(output_dir, 'budget_metrics.csv'), index=False)
    
    print(f"Sample data files saved to '{output_dir}' directory.")

if __name__ == "__main__":
    save_sample_data()