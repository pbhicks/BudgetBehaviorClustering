from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

from cluster import get_user_persona  # You’ll define this to run KMeans
from sample_data import get_sample_transactions  # Optional fallback

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@app.route('/')
def home():
    return jsonify({"message": "🚀 Budget Behavior Clustering API is live!"})

@app.route('/analyze', methods=['POST'])
def analyze_budget():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        df = pd.DataFrame(data['transactions'])
        income = data['income']
        savings_rate = data['savingsRate']
        
        persona, summary = get_user_persona(df, income, savings_rate)
        return jsonify({
            "persona": persona,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sample-data', methods=['GET'])
def sample_data():
    sample = get_sample_transactions()
    return jsonify(sample)

if __name__ == '__main__':
    app.run(debug=True)
