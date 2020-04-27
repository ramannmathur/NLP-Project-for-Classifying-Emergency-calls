# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:27:49 2020

@author: Raman
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__) #Initialize the flask App
model1 = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    vectorizer=pickle.load(open('vectorizer','rb'))
    int_features = request.form['dd']
    data = [int_features]
    pred = model1.predict(vectorizer.transform(data))
    output=pred
    return render_template('index1.html', prediction_text='Type of Emergency is {}'.format(output))

if __name__ == "__main__":
    app.run(port=5000,debug=True)

