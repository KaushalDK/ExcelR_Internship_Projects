#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Support Vector Machine Model for diabetic retinopathy')

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.number_input("Patient Age")
    systolic_bp = st.sidebar.number_input("systolic_bp")
    diastolic_bp = st.sidebar.number_input("diastolic_bp")
    cholesterol = st.sidebar.number_input("cholesterol")
    
    data = {'age':age,
            'systolic_bp':systolic_bp,
            'diastolic_bp':diastolic_bp,
            'cholesterol':cholesterol}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


loaded_model= load(open('trained_model.sav', 'rb'))


def retinopathy_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

prediction = retinopathy_prediction(df)

st.subheader('Predicted Result')
st.write("The person doesn't has retinopathy" if prediction[0] == 0 else "The person has retinopathy")








