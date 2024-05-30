from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('C:\\Users\\NAMO\\Desktop\\parkinsons_app\\parkinsons_model.pkl')
scaler = joblib.load('C:\\Users\\NAMO\\Desktop\\parkinsons_app\\scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        features = [float(x) for x in request.form.values()]
        # Convert to array
        features_array = np.array(features).reshape(1, -1)
        # Scale the input
        scaled_features = scaler.transform(features_array)
        # Predict using the model
        prediction = model.predict(scaled_features)
        # Return the result
        return render_template('index.html', prediction_text='Parkinson\'s Disease Likelihood: {}'.format('Yes' if prediction[0] else 'No'))

if __name__ == '__main__':
    app.run(debug=True)
