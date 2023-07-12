import streamlit as st
import time
import joblib
import sklearn
import catboost
import pandas
import numpy
import category_encoders

st.title("Loan Status")
st.write("This is a web app to predict whether a customer will claim the home loan or not.")

model = joblib.load('model.h5')
inputs = joblib.load('inputs.h5')

def predict(Gender, Married, Dependents, Education, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    test_df = pandas.DataFrame(columns=inputs)
    test_df.at[0, 'Gender'] = Gender
    test_df.at[0, 'Married'] = Married
    test_df.at[0, 'Dependents'] = Dependents
    test_df.at[0, 'Education'] = Education
    test_df.at[0, 'LoanAmount'] = LoanAmount
    test_df.at[0, 'Loan_Amount_Term'] = Loan_Amount_Term
    test_df.at[0, 'Credit_History'] = Credit_History
    test_df.at[0, 'Property_Area'] = Property_Area
    prediction = model.predict(test_df)
    return prediction[0]

def main():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.slider('Dependents', min_value=0, max_value=3, value=0, step=1)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    loan_amount = st.number_input('Loan Amount', min_value=1, max_value=600, value=1, step=1)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=12, max_value=480, value=12, step=12)
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    credit_history = st.checkbox('Credit History')
    results = predict(gender, married, dependents, education, loan_amount, loan_amount_term, credit_history, property_area)
    
    if st.button('Predict'):
        with st.spinner('Wait for it...'):
            time.sleep(1)
        label = ["not claim" , "claim"]
        st.write('The customer will', label[results], 'the home loan.')
    
if __name__ == '__main__':
    main()