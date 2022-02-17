import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
sns.set(style = 'darkgrid', font_scale = 6)

@st.cache
def load_data():
    return pd.read_csv('insurance_regression.csv')
    

data = load_data()

st.title('Insurance Pricing App')
st.write("""From the Data below we built a machine learning-based pricing model 
         to get quotation for each client based on their demmographics""")

st.sidebar.title("Insurance Pricing App")
st.sidebar.info("Change Parameter to see how insurance pricing change")
st.sidebar.title("Parameter")

age = st.sidebar.slider('Age', 0, 100, 24)
bmi = st.sidebar.slider('BMI', 13, 40, 31)
num_children = st.sidebar.slider('Number of Children', 0, 12, 1)

gender = st.sidebar.radio("Gender",("female","male"))

if gender =='male':
    is_female = 0
else:
    is_female = 1

smoker = st.sidebar.radio("Smoker??",("No","Yes"))

if smoker =='Yes':
    is_smoker = 1
else:
    is_smoker = 0
    
region = st.sidebar.selectbox("Region",['northwest','northeast','southeast','southwest'])

if region == 'northeast':
    loc_list = [1, 0, 0, 0]
elif region == 'northwest':
    loc_list = [0, 1, 0, 0]
elif region == 'southeast':
    loc_list = [0, 0, 1, 0]
elif region == 'southwest':
    loc_list = [0, 0, 0, 1]
                                         
st.subheader("Output Insurance Price")


filename = 'model.sav'
loaded_model = joblib.load(filename)

prediction = np.round(loaded_model.predict([[age, bmi, num_children, is_female, is_smoker] + loc_list])[0],2)
st.write(f"Suggested Insurance Price is: {prediction}")
