import streamlit as st
import pandas as pd
from sklearn. ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import plotly.express as px
import requests

# 페이지 설정 (탭 이름, 아이콘 등)
st.set_page_config(page_title="실시간 서버 모니터링",  # 원하는 이름으로 변경
                   page_icon="🚀",            # 이모지나 파일 경로
                   layout="wide"              # 레이아웃 설정(선택 사항)
)
st.title('🤖 서버 실시간 이상 탐지 대시보드 🤖')

st.info('이 앱은 머신러닝 모델을 활용하여 실시간으로 서버 상태를 모니터링할 수 있게 시각화한 대시보드를 제공합니다')

machine_num = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
                    '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
                    '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11']

# Machine selection
with st.sidebar:
  st.header('Monitoring Settings')

  selected_machine = st.sidebar.selectbox('대상 머신 선택', [f'machine-{i}' for i in machine_num])

# Column Rename (Data Preprocess)
df_train = pd.read_csv(f'https://raw.githubusercontent.com/roundy00/keroro-machinelearning/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{selected_machine}/{selected_machine}_test.csv')
df_input = pd.read_csv(f'https://raw.githubusercontent.com/roundy00/keroro-dashboard/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{selected_machine}/{selected_machine}_train.csv')

new_column_names = [
  'cpu_r', 'load_1', 'load_5', 'load_15', 'mem_shmem', 'mem_u', 'mem_u_e', 'total_mem',
  'disk_q', 'disk_r', 'disk_rb', 'disk_svc', 'disk_u', 'disk_w', 'disk_wa', 'disk_wb',
  'si', 'so', 'eth1_fi', 'eth1_fo', 'eth1_pi', 'eth1_po', 'tcp_tw', 'tcp_use',
  'active_opens', 'curr_estab', 'in_errs', 'in_segs', 'listen_overflows', 'out_rsts',
  'out_segs', 'passive_opens', 'retransegs', 'tcp_timeouts', 'udp_in_dg', 'udp_out_dg',
  'udp_rcv_buf_errs', 'udp_snd_buf_errs']
rename_dict = {f'col_{i}': new_column_names[i] for i in range(len(new_column_names))}

df_train.rename(columns=rename_dict, inplace=True)
df_input.rename(columns=rename_dict, inplace=True)

priority_columns = [
  'timestamp', 'cpu_r', 'load_1', 'load_5', 'mem_u',
  'disk_q', 'disk_r', 'disk_w', 'disk_u', 'eth1_fi', 'eth1_fo','tcp_timeouts']

priority_columns_train = priority_columns + ['label']
df_train = df_train[priority_columns_train]
df_input = df_input[priority_columns]

X = df_train.drop(labels = 'label', axis=1) # 학습-문제데이터
y = df_train.label # 학습-정답데이터


# Data Preparation : Model selection, time range setting
with st.sidebar:
  model_type = st.sidebar.radio('분석 모델 종류', ["ML (RandomForest)","ML (XGBoost)","DL (OmniAnomaly)", "DL (LSTM-NDT)", "DL (IMDiffusion)", "DL (Anomaly Transformer)", "DL (Pi-Transformer)"])
  time_range = st.select_slider('분석할 시간 범위', options = range(0, len(df_input)), value = (0,len(df_input)-1))

  scale_pos_weight = (len(y) - sum(y)) / sum(y)
  selected_model_dict = {"ML (RandomForest)" : RandomForestClassifier(class_weight='balanced',random_state = 42),
                         "ML (XGBoost)": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)}
  
# 슬라이더에서 선택된 범위만큼 데이터 자르기
display_df = df_input.iloc[time_range[0] : time_range[1] + 1]

# 메인 페이지에 현재 선택 정보 보여주기
selected_info = {'machine':selected_machine,
                 'model':model_type,
                 'start time':time_range[0],
                 'end time':time_range[1]}
input_info = pd.DataFrame([selected_info])
st.dataframe(input_info, hide_index=True)

# 1. 머신러닝 모델인 경우
if 'ML' in model_type:
  # 모델 학습
  model = selected_model_dict[model_type]
  model.fit(X, y)
  
  # 예측
  display_df['pred'] = model.predict(display_df)

# 예측값 시각화
st.write("### 🚨 이상 탐지 결과 (Prediction)")
pred_fig = px.line(display_df, x = 'timestamp', y = 'pred')
pred_fig.update_traces(line_color='#FF0000', line_width=2)
pred_fig.update_layout(
    xaxis=dict(fixedrange=True),
    yaxis=dict(fixedrange=True),
    dragmode=False
)

st.plotly_chart(pred_fig, use_container_width=True, config={'displayModeBar': False})

# 2. 딥러닝 모델인 경우 (API 호출)
elif "DL" in model_type:
    st.warning("⚠️ 딥러닝 모델은 서버로부터 실시간 분석 결과를 가져옵니다.")
    
    # 팀원에게 받은 서버 주소 적용
    API_URL = "https://unbarreled-uncrusted-juliana.ngrok-free.dev/predict" 
    
    # 실시간 업데이트를 위한 공간 확보
    chart_spot = st.empty()
    status_spot = st.empty()
    
    # 2. 100개씩 잘라서 실시간처럼 보여주는 루프
    # display_df에서 한 줄씩 서버로 쏘고 결과를 받아 업데이트합니다.
    results = []
    
    for i in range(len(display_df)):
        # 서버가 요구하는 38개 피처 추출 (timestamp 제외)
        # main.py의 ENC_IN=38 규격에 맞춤 
        current_row = display_df.iloc[i][new_column_names].tolist()
        
        try:
            # 🚨 POST 방식으로 'values' 키에 담아 전송 
            response = requests.post(API_URL, json={"values": current_row})
            
            if response.status_code == 200:
                res = response.json()
                
                # 결과값 저장 (서버가 아직 100개를 못 모았으면 False 반환됨) 
                results.append(1 if res['is_anomaly'] else 0)
                
                # 상태 표시
                if res['status'] == "collecting":
                    status_spot.info(f"데이터 수집 중... ({res['progress']})")
                else:
                    if res['is_anomaly']:
                        status_spot.error(f"🚨 이상 감지! Score: {res['score']:.4f}")
                    else:
                        status_spot.success(f"✅ 정상 상태 (Score: {res['score']:.4f})")
                
                # 3. 실시간 차트 업데이트 (최근 100개 결과 표시)
                if len(results) > 0:
                    with chart_spot.container():
                        # 현재까지의 결과를 반영한 임시 데이터프레임 생성
                        temp_df = display_df.iloc[:len(results)].copy()
                        temp_df['pred'] = results
                        
                        fig = px.line(temp_df.tail(100), x='timestamp', y='pred', title="Real-time Detection")
                        fig.update_traces(line_color='#FF0000')
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")

            else:
                st.error(f"서버 오류: {response.status_code}")
                break
                
        except Exception as e:
            st.error(f"연결 실패: {e}")
            break
            
        time.sleep(0.1) # 실시간 속도 조절

with st.expander('Data'):
  st.write('**Raw Data**')
  df_input

with st.expander('Feature visualization'):
    # 시각화할 컬럼들 리스트
    viz_cols = ['cpu_r', 'disk_r', 'mem_u', 'tcp_timeouts']
    
    for col in viz_cols:
        # 1. Plotly로 라인 차트 생성
        fig = px.line(display_df, x='timestamp', y=col, title=f'Server {col} Over Time')
        
        # 2. 상호작용(줌, 팬) 비활성화 설정
        fig.update_layout(
            xaxis=dict(fixedrange=True), # X축 고정
            yaxis=dict(fixedrange=True), # Y축 고정
            dragmode=False,               # 마우스 드래그 비활성화
            hovermode='x'                # 마우스를 올렸을 때 값만 보여줌
        )
        
        # 3. Streamlit에 출력 (config에서 도구 모음도 숨김)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
