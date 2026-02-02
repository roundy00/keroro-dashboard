import streamlit as st
import pandas as pd
from sklearn. ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import plotly.express as px
import requests
import time
import numpy as np
import shap

# 페이지 설정 (탭 이름, 아이콘 등)
st.set_page_config(page_title="실시간 서버 모니터링",  # 원하는 이름으로 변경
                   page_icon="🚀",            # 이모지나 파일 경로
                   layout="wide"              # 레이아웃 설정(선택 사항)
)
st.title('🤖 서버 실시간 이상 탐지 대시보드 🤖')

st.info('이 앱은 머신러닝과 딥러닝 모델을 활용하여 실시간으로 서버 상태를 모니터링할 수 있게 시각화한 대시보드를 제공합니다')

machine_num = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
                    '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
                    '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11']

# Machine selection
with st.sidebar:
  st.header('Monitoring Settings')

  selected_machine = st.sidebar.selectbox('대상 머신 선택', [f'machine-{i}' for i in machine_num])

# Column Rename (Data Preprocess)
df_train = pd.read_csv(f'https://raw.githubusercontent.com/roundy00/keroro-machinelearning/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{selected_machine}/{selected_machine}_train.csv')
df_input = pd.read_csv(f'https://raw.githubusercontent.com/roundy00/keroro-dashboard/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{selected_machine}/{selected_machine}_test.csv')

with st.sidebar:
  model_type = st.sidebar.radio('분석 모델 종류', ["ML (RandomForest)","ML (XGBoost)", "ML (IsolationForest)", "DL (OmniAnomaly)", "DL (LSTM-NDT)", "DL (IMDiffusion)", "DL (Anomaly Transformer)", "DL (Pi-Transformer)"])

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

with st.sidebar:
  time_range = st.select_slider('분석할 시간 범위', options = range(0, len(df_input)), value = (0,len(df_input)-1))
  
X = df_train.drop(columns = 'timestamp', axis=1) # 학습-문제데이터
# y = df_train.label # 학습-정답데이터

# scale_pos_weight = (len(y) - sum(y)) / sum(y)
selected_model_dict = {"ML (IsolationForest)": IsolationForest(contamination=0.01, random_state=42),
                       # "ML (RandomForest)" : RandomForestClassifier(class_weight='balanced',random_state = 42),
                       # "ML (XGBoost)": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42),
                       }


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

# 1. 머신러닝 모델인 경우

if 'ML' in model_type:
  model = selected_model_dict[model_type]
  
  # --- Isolation Forest 사전 탐지 파라미터 ---
  DANGER_LINE = 0.0    # 위험 기준 (이보다 낮으면 빨간색)
  WARNING_THRESHOLD = 0.05 # 사전 경보 기준 (주의 단계)
  WINDOW_SIZE = 5        # 점수 변화를 관찰할 윈도우
  
  if model_type == "ML (IsolationForest)":
    # 1. 학습은 그대로 진행 (정상 데이터 패턴 학습)
    model.fit(X)
    
    # 실시간 분석을 위한 공간
    status_box = st.empty()
    chart_box = st.empty()
    
    scores_history = []  # 점수 기록 저장용
    
    for i in range(len(display_df)):
      current_row = display_df.iloc[i:i+1][new_column_names]
      
      # [핵심] predict() 대신 decision_function() 사용 (수치화된 점수)
      # decision_function은 낮을수록(음수일수록) 더 이상함을 의미함
      score = model.decision_function(current_row)[0]
      scores_history.append(score)
      
      # 최근 N개의 점수 평균 계산
      recent_scores = scores_history[-WINDOW_SIZE:]
      avg_score = np.mean(recent_scores)
      
      # 2. 사전 탐지 로직 (Early Warning)
      is_anomaly = score < DANGER_LINE
      is_warning = (score < WARNING_THRESHOLD) and (not is_anomaly)
      
      # 3. 화면 표시
      with status_box.container():
        if is_anomaly:
          st.error(f"🚨 [위험] 시스템 장애 발생! (점수: {score:.4f})")
        elif is_warning:
          # 점수가 나빠지고 있는 상태
          st.warning(f"⚠️ [주의] 이상 전조 증상 포착! (점수: {score:.4f})")
        else:
          st.info(f"✅ [정상] 운영 상태 양호 (점수: {score:.4f})")

      # 4. 차트 업데이트 (실시간 점수 변화)
      with chart_box.container():
        plot_df = pd.DataFrame({
        'step': range(max(0, i-99), i+1),
        'score': scores_history[-100:]
        })
        
        fig = px.line(plot_df, x='step', y='score',
                      title=f"⚠️ {selected_machine} 사전 탐지 모니터링 (Score 기반)",
                      labels={'score': 'Anomaly Score', 'step': 'Time Step'})

        # [핵심] 노란색 '주의' 영역 표시 (Rectangles)
        # y0~y1 범위를 노란색 박스로 채워 전조 증상 구간을 시각화합니다.
        fig.add_hrect(y0=DANGER_LINE, y1=WARNING_LINE, 
                      fillcolor="yellow", opacity=0.3, line_width=0,
                      annotation_text="Warning Zone (Pre-detection)", 
                      annotation_position="top left")
    
        # [핵심] 빨간색 '위험' 영역 표시
        fig.add_hrect(y0=plot_df['score'].min() - 0.1, y1=DANGER_LINE, 
                      fillcolor="red", opacity=0.2, line_width=0,
                      annotation_text="Danger Zone", 
                      annotation_position="bottom left")
    
        # 기준선(Line) 추가
        fig.add_hline(y=DANGER_LINE, line_dash="dash", line_color="red")
        fig.add_hline(y=WARNING_LINE, line_dash="dot", line_color="orange")
        
        # y축 범위를 점수에 맞게 조정
        fig.update_layout(yaxis=dict(range=[-0.5, 0.5])) 
        st.plotly_chart(fig, use_container_width=True, key=f"pre_det_chart_{i}")

      time.sleep(0.05)
  
  # # [B] RandomForest / XGBoost (지도 학습) 처리
  # else:
  #   y = df_train['label'] # 지도학습에 필요한 라벨
  #   with st.spinner(f"[{selected_machine}] {model_type} 학습 중..."):
  #       model.fit(X, y)
  #   display_df['pred'] = model.predict(display_df[new_column_names])

  # # --- 시각화 섹션 ---
  # # 1. 이상 탐지 결과 알림 (깜빡이 효과)
  # if display_df['pred'].sum() > 0:
  #   trigger_alert_css()
  #   st.error(f"⚠️ 현재 범위 내에서 {int(display_df['pred'].sum())}건의 이상 징후가 포착되었습니다!")

  # # 2. 결과 그래프
  # st.write(f"### 🚨 이상 탐지 결과 ({model_type})")
  # pred_fig = px.line(display_df, x='timestamp', y='pred', title="Anomaly Detection Timeline")
  # pred_fig.update_traces(line_color='#FF0000', line_width=2)
  # st.plotly_chart(pred_fig, use_container_width=True)

  # 3. SHAP 원인 분석 (선택 사항)
  # Isolation Forest도 Tree 기반이라 TreeExplainer 사용 가능
  with st.spinner("판단 근거(SHAP) 분석 중..."):
    explainer = shap.TreeExplainer(model)
    # 속도를 위해 샘플링
    X_sample = display_df[new_column_names].sample(min(100, len(display_df)))
    shap_values = explainer.shap_values(X_sample)
    
    # Isolation Forest SHAP 대응
    if model_type == "ML (IsolationForest)":
      sv = shap_values 
    else:
      sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    importance = np.abs(sv).mean(axis=0)
    analysis_results = pd.DataFrame({'Feature': new_column_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    
    st.write("### 🔍 Root Cause Analysis (주요 원인 지표)")
    fig = px.bar(analysis_results.head(10)[::-1], x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
  
# ===============================================================================
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
            importance_dict = res_data["importance"]
            imp_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])

            # 3. 중요도 순으로 정렬 후 상위 10개 추출
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
        
            fig = px.bar(imp_df[::-1], x='Importance', y='Feature', orientation='h',
                         title=f"DL Model Analysis: {selected_machine}",
                         color_discrete_sequence=['#FF4B4B']) # DL은 강조색
            
            # 가독성을 위해 최상단에 큰 값이 오도록 정렬 유지
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
          st.error(f"서버 응답 실패 (Code: {response.status_code})")
      except Exception as e:
        st.error(f"코랩 서버 연결 실패: {e}")

  # DL일 때는 dynamic_features를 전체 피처로 설정하여 시각화 에러 방지
  dynamic_features = new_column_names

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
            st.session_state.last_anomaly_time = time.time()

          # 🚨 마지막 이상 감지 시점으로부터 30초 이내라면 계속 경고창 표시
          if time.time() - st.session_state.last_anomaly_time < 30:
              with alert_box.container():
                  trigger_alert_css()  # 빨간색 깜빡이 효과 유지
                  # 추가: 남은 시간 표시 (선택 사항)
                  remaining = int(30 - (time.time() - st.session_state.last_anomaly_time))
                  st.toast(f"🚨 이상 감지! 경고가 {remaining}초간 유지됩니다.")
          else:
              alert_box.empty()
          
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

# --- 공통 시각화 (모델이 선정한 주요 지표 Top 5 시각화) ---
with st.expander('🔍 Top 5 Influential Features Detail View'):
    st.write("모델이 분석한 이상 징후 기여도 상위 5개 지표의 변화 추이입니다.")
    
    # 1. 모델 종류(ML/DL)에 따라 상위 5개 컬럼 추출
    viz_cols = []
    
    if "ML" in model_type:
        if 'analysis_results' in locals():
            # ML 분석 결과에서 상위 5개 Feature 이름 가져오기
            viz_cols = analysis_results.head(5)['Feature'].tolist()
    
    elif "DL" in model_type:
        if 'imp_df' in locals():
            # DL 분석 결과에서 상위 5개 Feature 이름 가져오기
            viz_cols = imp_df.head(5)['Feature'].tolist()

    # 2. 추출된 컬럼이 있다면 시각화 수행
    if viz_cols:
        # 화면을 너무 길게 쓰지 않도록 2열 레이아웃 구성 (선택 사항)
        # col1, col2 = st.columns(2) 
        
        for idx, col in enumerate(viz_cols):
            # 시각화: 시계열 라인 차트
            fig = px.line(display_df, x='timestamp', y=col, 
                          title=f'📍 [Top {idx+1}] {col} Trend Analysis',
                          color_discrete_sequence=['#00CC96']) # 지표별 강조색
            
            # 차트 레이아웃 최적화 (가독성 향상)
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 모델 분석을 먼저 수행하면 주요 지표 그래프가 여기에 표시됩니다.")
