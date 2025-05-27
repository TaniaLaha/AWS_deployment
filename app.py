from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("model/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    prediction = model.predict([features])[0]

    iris_types = ['Setosa', 'Versicolor', 'Virginica']
    result = iris_types[prediction]
    return render_template('index.html', prediction_text=f'Predicted Iris Species: {result}')

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port='5000')