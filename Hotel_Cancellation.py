import numpy as np
import pandas as pd
import streamlit as st
import joblib 

# Lets Load all the Instances required over Here 
with open('transformer.joblib','rb') as file:
    transformer = joblib.load(file)

# Lets Load the model
    
with open('final_model.joblib','rb') as file:
    model = joblib.load(file)


st.title('INN Hotel Group')
st.header(':orange[This application will predict the chances of Booking Cancellations]')

# Lets Take input from user
amnth = st.slider('Select Your Month of Arrival', min_value=1,max_value=12)
wkd_lambda = (lambda x: 0 if x =='Monday'else
              1 if x=='Tuesday' else
              2 if x== 'Wednesday' else
              3 if x== ' Thursday' else
              4 if x== 'Friday' else
              5 if x== 'Saturday' else 6)
awkd = wkd_lambda(st.selectbox('Select Your Weekday of Arrival',['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']))
dwkd = wkd_lambda(st.selectbox('Select Your Weekday of Departure',['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']))

wkend = st.number_input('Enter How Many WeekEnd Nights are ther in Stay',min_value=0)
wk = st.number_input('Enter How Many Week Nights are ther in Stay',min_value=0)
totn = wkend + wk 
mkt = (lambda x: 0 if x=='Offline' else 1 )(st.selectbox('How Booking has been Made',['Online','Offline']))

lt = st.number_input('How many Days Prior the Booking was Made',min_value=0)
price = st.number_input('What is the Average Price Per Room',min_value=0)
adults = st.number_input('How Many Adult Members in Booking',min_value=0)
spcl = st.selectbox('Select the Number of Special Request Made',[0,1,2,3,4,5])
park = (lambda x: 0 if x=='No'else 1)(st.selectbox('Does Guest Need Parking Space',['Yes','No']))

# Lets Transform The Data 

lt_t,price_t = transformer.transform([[lt,price]])[0]

# Create the input list 
input_list = [lt_t,spcl,price_t,adults,wkend,park,wk,mkt,amnth,awkd,totn,dwkd]

# Make prediction 

prediction = model.predict_proba([input_list])[:, 1][0]

# Lets Show the Probability 
if st.button ('Predict'):
    st.success(f'Cancellation Chances {round(prediction,4)*100}%')