import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Data
data = {'Size': [20,5,15,18],
        'Price of House': [35000,20000,30000,32000]}
df = pd.DataFrame(data)

# Train
X = df[['Size']]
Y = df[['Price of House']]
model = LinearRegression()
model.fit(X,Y)

# UI
st.title("House Price Predictor")

size = st.number_input("Enter House Size:", min_value=1, step=1)
if st.button("Predict"):
    price = model.predict([[size]])[0][0]
    st.success(f"Predicted Price: {price:.2f}")
