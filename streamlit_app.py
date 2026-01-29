import streamlit as st
import pandas as pd

st.title('🤖 서버 실시간 이상 탐지 대시보드 🤖')

st.info('이 앱은 머신러닝 모델을 활용하여 실시간으로 서버 상태를 모니터링할 수 있게 시각화한 대시보드를 제공합니다')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/roundy00/keroro-machinelearning/refs/heads/master/Server-Machine-Dataset-main/processed_csv/machine-1-1/machine-1-1_test.csv')
  df

  st.write('**Feature**')
  X = df.drop('lable', axis=1)
  X

  st.write('**Anomaly  Detection**')
  y = df.label
  y
