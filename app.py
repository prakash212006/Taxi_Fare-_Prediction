import pickle
import numpy as np
import pandas as pd
import requests  # Import requests to get live exchange rates
from flask import Flask, request, jsonify, render_template

# Load the trained model and scaler
with open("trained_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Define numerical and categorical features
        num_features = ["trip_distance", "extra", "mta_tax", "tolls_amount", "imp_surcharge", "trip_duration"]
        cat_features = ["payment_type", "pickup_location_id", "dropoff_location_id"]

        # Convert input data to DataFrame
        df = pd.DataFrame([data])

        # Scale numerical features
        df[num_features] = scaler.transform(df[num_features])

        # Make prediction in USD
        prediction_usd = model.predict(df)[0]

        # Fetch the latest USD to INR exchange rate
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            exchange_rate = response.json()["rates"]["INR"]
        except Exception:
            exchange_rate = 83.0  # Use a fixed rate if API fails

        # Convert to INR
        prediction_inr = prediction_usd * exchange_rate

        # Return response w
        # ith ₹ symbol
        return jsonify({"predicted_fare_amount": f"₹{round(prediction_inr, 2)}"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

5
