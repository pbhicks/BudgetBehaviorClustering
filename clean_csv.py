import pandas as pd

# Clean 5-column sample data
data = [
    {"Income": 4000, "Housing": 1200, "Food": 450, "Entertainment": 300, "Savings Rate": 0.2},
    {"Income": 3500, "Housing": 900,  "Food": 350, "Entertainment": 250, "Savings Rate": 0.15},
    {"Income": 5000, "Housing": 1800, "Food": 500, "Entertainment": 400, "Savings Rate": 0.1},
    {"Income": 4200, "Housing": 1000, "Food": 400, "Entertainment": 300, "Savings Rate": 0.25},
    {"Income": 3000, "Housing": 800,  "Food": 300, "Entertainment": 200, "Savings Rate": 0.3},
]

df = pd.DataFrame(data)
df.to_csv("backend/data/Sample_Users_Budget_Profile.csv", index=False)
print("✅ Sample_Users_Budget_Profile.csv updated to 5 features.")
