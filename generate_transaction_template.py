import pandas as pd

data = [
    {"Date": "2024-01-01", "Category": "Housing", "Amount": 1200, "Description": "Rent payment"},
    {"Date": "2024-01-05", "Category": "Food", "Amount": 85.50, "Description": "Grocery store"},
    {"Date": "2024-01-10", "Category": "Transportation", "Amount": 60.00, "Description": "Gas station"},
    {"Date": "2024-01-15", "Category": "Entertainment", "Amount": 45.00, "Description": "Movie night"},
    {"Date": "2024-01-20", "Category": "Other", "Amount": 120.00, "Description": "Online shopping"}
]

df = pd.DataFrame(data)
df.to_csv("backend/data/Transaction_Template.csv", index=False)
print("✅ Transaction_Template.csv created!")
