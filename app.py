# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input from the form
            input_value = float(request.form['input_value'])

            # Make a prediction using the trained model
            prediction = model.predict([[input_value]])

            return render_template('index.html', prediction=f'Predicted Output: {prediction[0][0]:.2f}')

        except ValueError:
            return render_template('index.html', prediction='Invalid Input')

if __name__ == '__main__':
    app.run(debug=True)
