from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the pre-trained model
model = load_model('churn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        print("Received data:", data)
        
        # Extract the features array from the data
        features = np.array(data['features']).reshape(1, -1)
        print("Features:", features)
        
        # Make a prediction
        prediction = model.predict(features)
        print("Prediction:", prediction)
        
        # Return the prediction as a JSON response
        result = {"churn_probability": float(prediction[0])}
        return jsonify(result)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
