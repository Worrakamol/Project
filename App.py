# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:34:49 2025
@author: Mod
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# โหลด model และ encoders
@st.cache_resource
def load_model():
    model = pickle.load(open('loan_approval_model.pkl', 'rb'))
    status_encoder = pickle.load(open('employment_encoder.pkl', 'rb'))
    approval_encoder = pickle.load(open('approval_encoder.pkl', 'rb'))
    try:
        scaler = pickle.load(open('scaler.pkl', 'rb'))
    except FileNotFoundError:
        scaler = None
    return model, status_encoder, approval_encoder, scaler

model, status_encoder, approval_encoder, scaler = load_model()

# ส่วนหัวเว็บ
st.title('🔎 Loan Approval Prediction')
st.write('กรุณากรอกข้อมูลเพื่อทำนายผลการอนุมัติสินเชื่อ')

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
        # เตรียม input เป็น numpy array
        input_array = np.array([[
            income,
            credit_score,
            loan_amount,
            dti_ratio,
            status_encoder.transform([employment_status])[0]
        ]])

        # ถ้ามี scaler
        if scaler:
            input_array = scaler.transform(input_array)

        # ทำนายผล
        try:
            prediction = model.predict(input_array)
            result = approval_encoder.inverse_transform(prediction)[0]

            # แสดงผลลัพธ์
            if result == 'Approved':
                st.success(f'✅ ผลการทำนาย: {result} (ผ่านการอนุมัติ)')
            else:
                st.error(f'❌ ผลการทำนาย: {result} (ไม่ผ่านการอนุมัติ)')

        except Exception as e:
            st.error(f'เกิดข้อผิดพลาดในการทำนาย: {e}')
