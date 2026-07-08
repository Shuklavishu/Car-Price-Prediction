import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import streamlit as st

def main():
   html_temp = """<h1>Car Price Prediction App</h1>"""
   model = xgb.XGBRegressor()
   model.load_model("xgb_model.json")

   st.markdown(html_temp, unsafe_allow_html=True)
   st.markdown("This app predicts the price of a car based on its features.") 

   present_price = st.number_input("Please enter ex-Showroom Price of the car (in lakh):", 2.5,25.0, step=1.0)
   kms_driven = st.number_input("Please enter the distance driven by the car (in km):", 100, 500000, step=100)
   
   fuel_type = st.selectbox("Please select the fuel type", ["Petrol", "Diesel", "CNG"])
   fuel_type_value = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
   
   seller_type = st.selectbox("Please select the seller type", ["Dealer", "Individual"])
   seller_type_value = {"Dealer": 0, "Individual": 1}[seller_type]
   
   transmission = st.selectbox("Please select the transmission type", ["Manual", "Automatic"])
   transmission_value = {"Manual": 0, "Automatic": 1}[transmission]
   
   owner_type = st.selectbox("Please select the owner type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
   owner_value = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, "Fourth & Above Owner": 3}[owner_type]
   
   car_age = st.slider("How old is the car? (years)", 0, 25, 1)
   
   data_new = pd.DataFrame({
         'Present_Price': [present_price],
         'Kms_Driven': [kms_driven],
         'Fuel_Type': [fuel_type_value],
         'Seller_Type': [seller_type_value],
         'Transmission': [transmission_value],
         'Owner': [owner_value],
         'Age': [car_age]
   })
   

   if st.button("Predict"):
       pred = model.predict(data_new)
       st.success("you can sell the car at {:.2f} lakh".format(pred[0]))
   
   
if __name__=='__main__':
    main()
