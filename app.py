# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('models/random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    try:
        # Convert the JSON into a DataFrame with expected columns
        input_data = pd.DataFrame([data])


        # Make predictions
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Prepare and send response
        response = {
            'prediction': int(prediction[0]),
            'probability': prediction_proba[0].tolist()
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
