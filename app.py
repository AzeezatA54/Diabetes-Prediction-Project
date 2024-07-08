import numpy as np
import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open("DiabetesModel.pkl", "rb"))


def prediction(input_data):
    input_to_array= np.asarray(input_data)
    input_array_reshaped = input_to_array.reshape(1,-1)
    
    prediction =model.predict(input_array_reshaped)
    
    return prediction
    
    if (prediction[0] ==0):
        return('You are not Diabetic')
    else:
        return('You are Diabetic')
    

def main():
    st.title("Diabetes Prediction App")
    html_template = """
    <div style="background-color:pink; padding:20px; border-radius: 10px;">
    <h2 style="color:white;text-align:center;">Diabetes Prediction App</h2>
    </div>
    """

    st.markdown(html_template, unsafe_allow_html=True)
    Pregnancies=st.text_input("No of Pregnancies")
    Glucose=st.text_input("Glucose Concentration")
    BloodPressure=st.text_input("Blood Pressure")
    SkinThickness=st.text_input("Skin Thickness")
    Insulin=st.text_input("Insulin")
    BMI=st.text_input("Body Mass Index")
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function")
    Age=st.text_input("Age")
    
    diagnosis = " "

    if st.button("Diabetes Prediction Result"):

        diagnosis = prediction([Pregnancies, Glucose,BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)



if __name__ == "__main__":
    main()