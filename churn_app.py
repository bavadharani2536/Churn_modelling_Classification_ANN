import pandas as pd
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#laoding the trained model
model=load_model('Churning.h5')

#loading the scaler and encoder pickles files

with open('Label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('One_hot_ENcoder.pkl','rb') as file:
    oh_encoder=pickle.load(file)

with open('Standard_Scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app
st.title('CUSTOMER CHURN PREDCITION')

# User input
geography = st.selectbox('Geography', oh_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#Data Transformation
def transformed_data(data):
    encoded_cols = oh_encoder.transform([[geography]]).toarray()
    encoded_data = pd.DataFrame(encoded_cols, columns=oh_encoder.get_feature_names_out(['Geography']))
    df=pd.concat([data,encoded_data],axis=1)

    scaled_df=scaler.transform(df)

    return scaled_df

#MOdel Prediciton
def model_prediction(df):
    prediction=model.predict(transformed_data(df))
    return 'will Churn-OOPS :(' if prediction[0][0]>0.5 else 'Will not churn-Yaay!!!'

if st.button('Predict'):

    st.write(f'Customer {model_prediction(input_data)}')