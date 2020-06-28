import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump, load

clf = load('model.joblib')
target_names = np.array(['setosa', 'versicolor', 'virginica'])

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4., 8., 6.)
    sepal_width = st.sidebar.slider('Sepal width', 2., 4.5, 3.3)
    petal_length = st.sidebar.slider('Petal length', 1., 7., 4.)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.)
    data = data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(target_names)

st.subheader('Prediction')
st.write(target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)