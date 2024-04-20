from flask import Flask, request,render_template, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression  # Example model - replace with your model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
app = Flask(__name__)

data=pd.read_csv("Iris.csv")
mapspecies={'Iris-setosa':0, 'Iris-versicolor':1,'Iris-virginica':2}
data['Species']=data['Species'].map(mapspecies)
x=data[['SepalWidthCm','SepalLengthCm','PetalLengthCm','PetalWidthCm']].values
y=data[['Species']].values
Reshaped_y=y.reshape(150,)
print(x[:5])
print(y[:5])

#Logistic regression
from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression()
model_lr.fit(x,y)
expected=y
predicted = model_lr.predict(x)
print(predicted)

# Define your machine learning model
#model = LinearRegression()  # Example model - replace with your model
#model.fit(X_train, y_train)  # Example training step - replace with your training code
# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to accept input data and return predictions
@app.route('/predict', methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction
    prediction = model_lr.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Convert prediction to human-readable species
    species = 'Setosa' if prediction == 0 else 'Versicolor' if prediction == 1 else 'Virginica'

    # Render the result page with the prediction
    return render_template('result.html', species=species)
    
# Function to preprocess input data
def preprocess_input_data(data):
    # Preprocess the input data here as needed
    # For example, convert it to a numpy array
    processed_data = np.array(data)

    return processed_data

if __name__ == '__main__':
    app.run(debug=True)
