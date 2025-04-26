# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:34:49 2025

@author: Mod
"""

import streamlit as st
import pandas as pd
import pickle

# โหลด model และ encoders
model = pickle.load(open('loan_approval_model.pkl', 'rb'))
status_encoder = pickle.load(open('employment_encoder.pkl', 'rb'))
approval_encoder = pickle.load(open('approval_encoder.pkl', 'rb'))

# ส่วนหัวเว็บ
st.title('Loan Approval Prediction')
st.write('กรอกข้อมูลเพื่อทำนายผลการอนุมัติสินเชื่อ')

# Input form
income = st.number_input('รายได้ต่อปี (Income)', min_value=0)
credit_score = st.number_input('คะแนนเครดิต (Credit Score)', min_value=0, max_value=850)
loan_amount = st.number_input('จำนวนเงินที่ต้องการกู้ (Loan Amount)', min_value=0)
dti_ratio = st.number_input('Debt-to-Income Ratio (%)', min_value=0.0, max_value=100.0)
employment_status = st.selectbox('สถานะการทำงาน', status_encoder.classes_)

if st.button('ทำนายผลการอนุมัติ'):
    input_data = pd.DataFrame({
        'Income': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'DTI_Ratio': [dti_ratio],
        'Employment_Status': [status_encoder.transform([employment_status])[0]]
    })

    prediction = model.predict(input_data)
    result = approval_encoder.inverse_transform(prediction)[0]

    st.success(f'ผลการทำนาย: {result}')
