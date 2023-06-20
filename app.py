import streamlit as st
import pickle
import pandas as pd
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
data = pickle.load(open('dataset.pkl','rb'))

st.title("Laptop Price Predictor")
st.markdown("<h1 style='font-size:24px;'>(SMARD Team)</h1>", unsafe_allow_html=True)

company = st.selectbox('Brand', data['Company'].unique())

# type of Laptop
type = st.selectbox('Type', data['TypeName'].unique())

# RAM present in Laptop
ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])

# OS of Laptop
operatingsystem = st.selectbox('OS', data['OpSys'].unique())

# weight of Laptop
weight = st.number_input('Weight of the Laptop')

# screen size (inches)
inches = st.number_input('Screen Size (Inches)')

# CPU
cpu = st.selectbox('CPU', data['CPU'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

if st.button('Predict Price'):

    query = np.array([company,type,ram,operatingsystem,weight,inches,cpu,hdd,ssd])
    
    query = query.reshape(1,9)
    
    st.title("Predicted price for this laptop coulde be between " + str(int(np.exp(pipe.predict(query)[0]))))

    prediction = int(np.exp(pipe.predict(query)[0]))

    st.title("Predicted price for this laptop could be between " +
             str(prediction - (prediction * 0.16)) + "euro to" +
             str(prediction + (prediction * 0.16)) + "euro")