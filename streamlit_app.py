import streamlit as st
import pandas as pd

st.title('🤖 서버 실시간 이상 탐지 대시보드 🤖')

st.info('이 앱은 머신러닝 모델을 활용하여 실시간으로 서버 상태를 모니터링할 수 있게 시각화한 대시보드를 제공합니다')

with st.expander('Data'):
  st.write('**Raw Data**')

  # 데이터 로드 & 컬럼명 수정
  df = pd.read_csv('https://raw.githubusercontent.com/roundy00/keroro-machinelearning/refs/heads/master/Server-Machine-Dataset-main/processed_csv/machine-1-1/machine-1-1_test.csv')
  new_column_names = [
    'cpu_r', 'load_1', 'load_5', 'load_15', 'mem_shmem', 'mem_u', 'mem_u_e', 'total_mem',
    'disk_q', 'disk_r', 'disk_rb', 'disk_svc', 'disk_u', 'disk_w', 'disk_wa', 'disk_wb',
    'si', 'so', 'eth1_fi', 'eth1_fo', 'eth1_pi', 'eth1_po', 'tcp_tw', 'tcp_use',
    'active_opens', 'curr_estab', 'in_errs', 'in_segs', 'listen_overflows', 'out_rsts',
    'out_segs', 'passive_opens', 'retransegs', 'tcp_timeouts', 'udp_in_dg', 'udp_out_dg',
    'udp_rcv_buf_errs', 'udp_snd_buf_errs']
  rename_dict = {f'col_{i}': new_column_names[i] for i in range(len(new_column_names))}
  df.rename(columns=rename_dict, inplace=True)
  priority_columns = [
    'timestamp', 'cpu_r', 'load_1', 'load_5', 'mem_u',
    'disk_q', 'disk_r', 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo','tcp_timeouts']

  priority_columns_test = priority_columns + ['label']
  df = df[priority_columns_test]
  df

  st.write('**Feature**')
  X = df.drop(labels = 'label', axis=1)
  X
  y = df.label

with st.expander('Data visualization'):
  st.line_chart(data=df, x='timestamp', y='cpu_r')

# Data Preparation
with st.sidebar:
  st.header('Input Features')

  machine_num = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
                    '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
                    '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11']
  selected_machine = st.sidebar.selectbox('대상 머신 선택', [f'machine-{i}' for i in machine_num])
  time_range = st.select_slider('분석할 시간 범위', options = range(0, len(df)), value = (0,len(df)))
  
