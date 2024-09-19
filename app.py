import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('/Users/gadgetspro/Desktop/village_healthcare_model.pkl','rb') as file:
    model = pickle.load(file)

# Title of the web app
st.title("Village Healthcare Access Prediction")

# Add inputs for user to fill
population = st.number_input("Enter Population", min_value=0)
distance_hospital = st.number_input("Enter Distance to Nearest Hospital (in km)", min_value=0)
health_centers = st.number_input("Enter Number of Health Centers", min_value=0)
doctors = st.number_input("Enter Number of Doctors", min_value=0)
nurses = st.number_input("Enter Number of Nurses", min_value=0)

# When the user clicks the 'Predict' button
if st.button("Predict Policy Impact Score"):
    # Create an input array for the model
    input_data = np.array([[population, distance_hospital, health_centers, doctors, nurses]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the result
    st.success(f"Predicted Policy Impact Score: {prediction[0]}")

