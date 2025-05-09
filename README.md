# 💸 Budget Behavior Clustering

A modular Streamlit application built to help users understand and optimize their financial behavior. Using unsupervised machine learning (KMeans clustering), this tool classifies users into financial personas based on their spending patterns—empowering better budgeting decisions through data.

Developed as part of a graduate finance capstone at the University of Arkansas, this app supports both direct budget entry and transaction data upload, with interactive visual feedback and tailored recommendations.

## 🧠 Overview

Budget Behavior Clustering uses machine learning techniques to analyze spending and saving patterns. By clustering users into distinct financial personas, the application offers personalized insights and recommendations to improve financial health.

This project blends machine learning with personal finance UX:

* **KMeans Clustering** to identify user personas
* **Streamlit UI** with dual input modes (Manual + File Upload)
* **Plotly Visualizations** for clean, modern insights
* **Sample Data** for instant testing and demo use
* **Modular Python Architecture** for clarity and extensibility

## ✨ Key Features

- **Budget Analysis**: Enter your income and expense data to receive a detailed analysis
- **Financial Persona Identification**: Discover your financial behavior pattern based on clustering algorithms
- **Personalized Recommendations**: Get tailored recommendations based on your financial persona
- **Interactive Visualizations**: Explore your budget data through multiple visualization types
- **Financial Health Score**: Receive a score and rating of your overall financial health

### Clustering Logic

* 🧠 **Persona Identification**
  Groups users into:

  * Cautious Saver
  * Balanced Spender
  * Aggressive Investor

* 🔍 **KMeans Algorithm**
  Clustering based on budget category percentages + savings rate

### Dual Input Modes

* 📝 **Manual Budget Input**
  Users enter income, essential, discretionary spending, and savings rate

* 📁 **Transaction Analysis**
  Upload .csv/.xlsx files with real transaction data (with instructions and sample)

## 🔧 Project Structure

```
BudgetBehaviorClustering/
├── backend/                  # Backend Python code
│   ├── data/                 # Data files
│   ├── models/               # Trained models
│   ├── logs/                 # Application logs
│   ├── app.py                # Streamlit interface
│   ├── cluster.py            # Clustering algorithms
│   ├── sample_data.py        # Sample data generator
│   ├── utils.py              # Utility functions
│   └── visuals.py            # Visualization functions
├── frontend/                 # React frontend (optional)
│   └── BudgetBehaviorClusteringReact/
│       ├── public/           # Public assets
│       └── src/              # React source code
├── main.py                   # Main application entry point
└── requirements.txt          # Python dependencies
```

## ⚙️ Installation

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/pbhicks/BudgetBehaviorClustering.git
   cd BudgetBehaviorClustering
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # OR
   .venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Application

1. Run the main application:
   ```bash
   python main.py
   ```

2. Generate fresh sample data (optional):
   ```bash
   python main.py --generate-data
   ```

3. Retrain the clustering model (optional):
   ```bash
   python main.py --train-model
   ```

4. Open the Streamlit interface in your browser (it should open automatically), or go to:
   ```
   http://localhost:8501
   ```

## 💼 How to Use

### 🧾 Manual Budget Input

1. Enter your **monthly income**
2. Input your budget information in the sidebar:
   - Housing expenses
   - Food expenses
   - Entertainment expenses 
   - Savings rate (%)
3. Click **Analyze My Budget** to see your results
4. Explore the different visualization tabs to understand your financial persona better

### 📂 Transaction Analysis

* Upload a file with:
  * `Date`, `Category`, `Amount`, `Description`
* Use the sidebar to select months and view spending trends
* The app calculates your financial persona and spending breakdown

## 💼 Financial Personas

The application identifies three main financial personas:

| Persona                | Traits                                                                |
| ---------------------- | --------------------------------------------------------------------- |
| 💰 **Cautious Saver**  | High savings, minimal lifestyle spending, essential-focused budgeting |
| ⚖️ **Balanced Spender**| Steady mix of spending and saving, balanced categories                |
| 📈 **Aggressive Investor** | Risk-tolerant, higher lifestyle allocation, investment-driven goals|

## 📚 Dependencies

This app uses:
* `streamlit`
* `pandas`
* `numpy`
* `plotly`
* `scikit-learn`
* `joblib`
* `matplotlib`

## 👤 Author

**Payton Hicks**  
MS Finance Candidate – University of Arkansas  
Cost Accountant | Builder of Tools | Advocate for Financial Clarity  
📍 GitHub: [pbhicks](https://github.com/pbhicks)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.