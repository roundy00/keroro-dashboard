import streamlit as st
import pandas as pd
from sklearn. ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import plotly.express as px
import requests
import time

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

with st.sidebar:
  model_type = st.sidebar.radio('분석 모델 종류', ["ML (RandomForest)","ML (XGBoost)","DL (OmniAnomaly)", "DL (LSTM-NDT)", "DL (IMDiffusion)", "DL (Anomaly Transformer)", "DL (Pi-Transformer)"])

if 'ML' in model_type:
  df_train = df_train[priority_columns_train]
  df_input = df_input[priority_columns]
else:
  pass

with st.sidebar:
  time_range = st.select_slider('분석할 시간 범위', options = range(0, len(df_input)), value = (0,len(df_input)-1))
  
X = df_train.drop(labels = 'label', axis=1) # 학습-문제데이터
y = df_train.label # 학습-정답데이터

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

# ==========================================
# 경고 발생 시 화면을 붉은색으로 깜빡이게 만드는 CSS입니다.
def trigger_alert_css():
    st.markdown(
        """
        <style>
        @keyframes blinker {
            50% { opacity: 0; }
        }
        .emergency-alert {
            background-color: #FF0000;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            border-radius: 10px;
            animation: blinker 0.5s linear infinite;
            margin-bottom: 20px;
        }
        </style>
        <div class="emergency-alert">
            🚨 EMERGENCY: ANOMALY DETECTED 🚨
        </div>
        """,
        unsafe_allow_html=True
    )
# ===============================================

# --- 1. 머신러닝(ML) 모델인 경우 ---
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

# --- ML용 SHAP 분석 추가 ---
  st.write("### 🔍 Root Cause Analysis (SHAP for ML)")
  if st.button("ML 모델 원인 분석 시작"):
      with st.spinner("SHAP 값을 계산 중입니다..."):
          import shap
          # ML은 데이터가 작으므로 마지막 100개 데이터로 로컬에서 직접 계산
          explainer = shap.TreeExplainer(model)
          shap_values = explainer.shap_values(display_df[priority_columns[1:]].iloc[-100:])
          
          # 클래스가 2개(정상/이상)인 경우 '이상(1)'에 대한 기여도 추출
          if isinstance(shap_values, list): # RF/XGB 특성상 리스트로 나올 수 있음
              sv = shap_values[1] 
          else:
              sv = shap_values

          importance = np.abs(sv).mean(axis=0)
          imp_df = pd.DataFrame({
              'Feature': priority_columns[1:],
              'Importance': importance
          }).sort_values(by='Importance', ascending=False).head(15)

          fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="ML Feature Importance")
          st.plotly_chart(fig)


# 2. 딥러닝 모델인 경우 (API 호출)
elif "DL" in model_type:
    st.write("### 🧠 Deep Learning Root Cause Analysis")
    if st.button("코랩 서버에 분석 요청"):
        # DL은 38개 전체 피처를 사용 (new_column_names)
        target_data = display_df[new_column_names].iloc[-100:].values.tolist()
        
        payload = {
            "machine_name": selected_machine,
            "window": target_data
        }
        
        with st.spinner(f"코랩 GPU에서 {selected_machine} 분석 중..."):
            try:
                # 본인의 최신 ngrok 주소로 수정 필수
                COLAB_URL = "https://nontractable-hailey-petiolar.ngrok-free.dev/analyze"
                response = requests.post(COLAB_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    res_data = response.json()
                    if "error" in res_data:
                        st.error(f"서버 에러: {res_data['error']}")
                    else:
                        importance = res_data["importance"]
                        imp_df = pd.DataFrame({
                            'Feature': new_column_names,
                            'Importance': importance
                        }).sort_values(by='Importance', ascending=False).head(15)
                        
                        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                     title=f"DL Model Analysis: {selected_machine}",
                                     color_discrete_sequence=['#FF4B4B']) # DL은 강조색
                        st.plotly_chart(fig)
                else:
                    st.error(f"서버 응답 실패 (Code: {response.status_code})")
            except Exception as e:
                st.error(f"코랩 서버 연결 실패: {e}")
  
    st.warning("⚠️ 딥러닝 모델은 서버로부터 실시간 분석 결과를 가져옵니다.")
    
    # 팀원에게 받은 서버 주소 적용
    API_URL = "https://unbarreled-uncrusted-juliana.ngrok-free.dev/predict" 
    
    # 실시간 대시보드 구성을 위한 공간
    status_box = st.empty()
    alert_box = st.empty()  # 경고창 전용 공간
    chart_box = st.empty()
    
    # 결과 저장 리스트
    scores = []
    
    # 시뮬레이션 시작 (test_client.py의 로직을 Streamlit 안으로 가져옴)
    # display_df의 데이터를 한 줄씩 쏘며 결과를 받아옵니다.
    for i in range(len(display_df)):
        # 38개 피처 추출 (new_column_names 활용)
        current_row = display_df.iloc[i][new_column_names].values.tolist()
        
        try:
            # 🚨 서버에 현재 행 데이터를 보내고 결과를 받음 
            response = requests.post(API_URL, json={"values": current_row})
            
            if response.status_code == 200:
                res = response.json()
                
                if res['status'] == "ready":
                    # 결과값 업데이트
                    is_anomaly = res['is_anomaly']
                    score = res['score']
                    scores.append(score)

                    # 🚨 이상 감지 시 '개요란한' 경고 발생
                    if is_anomaly:
                        with alert_box.container():
                            trigger_alert_css()  # 빨간색 깜빡이 효과
                            st.snow()            # 눈 내리는 효과 추가 (옵션)
                    else:
                        alert_box.empty()        # 정상일 때는 경고창 제거
                    
                    # 상태 업데이트
                    with status_box.container():
                        # 서버에서 온 실제 threshold와 score를 직접 텍스트로 찍어봅니다.
                        current_threshold = res.get('threshold', 0.5)
                        current_score = res.get('score', 0.0)
                        
                        if is_anomaly:
                            st.error(f"🚨 이상 발생! 점수: {current_score:.6f} (임계치: {current_threshold:.6f})")
                        else:
                            # 점수가 임계치에 얼마나 근접했는지 보여줍니다.
                            st.info(f"✅ 정상 (점수: {current_score:.6f} / 임계치: {current_threshold:.6f})")
                    
                    # 차트 업데이트 (최근 100개 데이터)
                    with chart_box.container():
                        latest_scores = scores[-100:]
                        temp_df = pd.DataFrame({
                            'step': range(len(scores) - len(latest_scores), len(scores)),
                            'score': latest_scores
                        })
                        fig = px.line(temp_df, x='step', y='score', title="Real-time Anomaly Score (Last 100)")
                        # 임계치 선 추가 (main.py의 THRESHOLD 사용) 
                        fig.add_hline(y=res['threshold'], line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True, key=f"dl_chart_{i}")
                
                else:
                    # 데이터 수집 단계 (WIN_SIZE 100개 채우는 중) 
                    status_box.info(f"⏳ 서버 데이터 축적 중... ({res['progress']})")
            
            else:
                st.error(f"서버 오류: {response.status_code}")
                break
                
        except Exception as e:
            st.error(f"연결 실패: {e}")
            break
            
        time.sleep(0.2) # test_client.py의 전송 속도와 맞춤


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
