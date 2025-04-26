# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:34:49 2025
@author: Mod
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_model():
    with open('loan_approval_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('employment_status_encoder.pkl', 'rb') as f:
        status_encoder = pickle.load(f)
    with open('approval_encoder.pkl', 'rb') as f:
        approval_encoder = pickle.load(f)
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
    return model, status_encoder, approval_encoder, scaler

model, status_encoder, approval_encoder, scaler = load_model()


st.title('🔎 Loan Approval Prediction')
st.write('Please enter your information to predict your loan approval status.')

# Input form
with st.form('input_form'):
    income = st.number_input('💰 รายได้ต่อปี (Income)', min_value=0, format='%d')
    credit_score = st.number_input('📈 คะแนนเครดิต (Credit Score)', min_value=0, max_value=850, format='%d')
    loan_amount = st.number_input('🏦 จำนวนเงินที่ต้องการกู้ (Loan Amount)', min_value=0, format='%d')
    dti_ratio = st.number_input('📊 Debt-to-Income Ratio (%)', min_value=0.0, max_value=100.0, format='%f')
    employment_status = st.selectbox('🧑‍💼 สถานะการทำงาน', status_encoder.classes_)

    submitted = st.form_submit_button('ทำนายผลการอนุมัติ')

if submitted:
    if income == 0 or credit_score == 0 or loan_amount == 0:
        st.warning('⚠️ กรุณากรอกข้อมูลให้ครบถ้วนก่อนทำการทำนาย')
    else:
        try:
            
            input_array = np.array([[ 
                income,
                credit_score,
                loan_amount,
                dti_ratio,
                status_encoder.transform([employment_status])[0]
            ]])

            
            if scaler:
                input_array = scaler.transform(input_array)

            
            prediction = model.predict(input_array)
            result = approval_encoder.inverse_transform(prediction)[0]

            if result == 'Approved':
                st.success(f'✅ ผลการทำนาย: {result} (ผ่านการอนุมัติ)')
            else:
                st.error(f'❌ ผลการทำนาย: {result} (ไม่ผ่านการอนุมัติ)')

        except Exception as e:
            st.error(f'เกิดข้อผิดพลาดในการทำนาย: {e}')
