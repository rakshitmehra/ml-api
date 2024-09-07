from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

try:
    model = joblib.load('./SavedModels/ml_model.pkl')
    label_encoder = joblib.load('./SavedModels/label_encoder.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    
@app.route('/')
def home():
    return "<h1> Machine Learning Model is Running!</h1>"

@app.route('/', methods=['POST'])
def predict():
    if not model or not label_encoder:
        return jsonify({"error": "Model or encoder not loaded"}), 500
    
    try:
        data = request.json
        results = []
        
        for item in data:
            last_event_category = label_encoder.transform([item['last_event_category']])[0]
            second_last_event_category = label_encoder.transform([item['second_last_event_category']])[0]
            most_frequent_category = label_encoder.transform([item['most_frequent_category']])[0]
            total_attended_events = item['total_attended_events']
            
            features = np.array([[last_event_category, second_last_event_category, most_frequent_category, total_attended_events]])
            
            prediction = model.predict(features)
            predicted_category = label_encoder.inverse_transform(prediction)[0]
            
            result = {
                'id': item['id'],
                'email': item['email'],
                'last_event_category': item['last_event_category'],
                'second_last_event_category': item['second_last_event_category'],
                'most_frequent_category': item['most_frequent_category'],
                'total_attended_events': item['total_attended_events'],
                'prediction': predicted_category
            }
            
            results.append(result)
        
        return jsonify(results)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
