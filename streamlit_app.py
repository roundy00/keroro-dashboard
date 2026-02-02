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

# 경고 상태를 관리하기 위한 변수 초기화
if 'mute_alert' not in st.session_state:
  st.session_state.mute_alert = False
if 'current_idx' not in st.session_state:
  st.session_state.current_idx = 0

# --- [추가] SHAP 분석 주기를 관리하기 위한 세션 변수 ---
if 'last_shap_time' not in st.session_state:
  st.session_state.last_shap_time = 0

# 자동 분석 활성화 체크박스 (사이드바 메뉴들 사이에 추가하세요)
auto_dl_analyze = st.sidebar.checkbox("실시간 DL 원인 분석 활성화", value=True)

DETECTION_URL = "https://unbarreled-uncrusted-juliana.ngrok-free.dev/predict"
SHAP_URL = "https://unbarreled-uncrusted-juliana.ngrok-free.dev/analyze"

# Machine selection
with st.sidebar:
  st.header("Emergency Control")
  
  if st.button("🚨 경보 강제 종료 (Mute)"):
    st.session_state.mute_alert = True
    st.success("경보가 일시 중지되었습니다.")
  
  if st.button("🔄 분석 처음부터 다시 시작"):
    st.session_state.current_idx = 0  # 인덱스 초기화
    st.rerun()  # 페이지 새로고침
      
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
  
X = df_train.drop(columns = 'timestamp', axis=1) # 학습-문제데이터
# y = df_train.label # 학습-정답데이터

# scale_pos_weight = (len(y) - sum(y)) / sum(y)
selected_model_dict = {"ML (IsolationForest)": IsolationForest(contamination=0.01, random_state=42),
                       # "ML (RandomForest)" : RandomForestClassifier(class_weight='balanced',random_state = 42),
                       # "ML (XGBoost)": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42),
                       }

with st.sidebar:
  start_point = st.number_input('시작 인덱스 설정', min_value=0, max_value=len(df_input)-1, value=0)
# 슬라이더에서 선택된 범위만큼 데이터 자르기
display_df = df_input.iloc[start_point:]

# 메인 페이지에 현재 선택 정보 보여주기
selected_info = {'Machine':selected_machine,
                 'Model':model_type,
                 'Status': 'Running'}
input_info = pd.DataFrame([selected_info])
st.dataframe(input_info, hide_index=True)

# ==========================================
# 경고 발생 시 화면을 붉은색으로 깜빡이게 만드는 CSS입니다.
def trigger_emergency_alert():
  # 화면 전체 배경을 붉게 만들고 흔드는 애니메이션
  st.markdown(
      """
      <style>
      @keyframes shake {
          0% { transform: translate(1px, 1px) rotate(0deg); }
          10% { transform: translate(-1px, -2px) rotate(-1deg); }
          30% { transform: translate(3px, 2px) rotate(0deg); }
          50% { transform: translate(-1px, 2px) rotate(1deg); }
          100% { transform: translate(1px, -2px) rotate(-1deg); }
      }
      .stApp {
          animation: shake 0.5s infinite;
          background-color: #440000 !important;
      }
      .extreme-alert {
          background-color: #FF0000;
          color: yellow;
          padding: 30px;
          text-align: center;
          font-size: 50px;
          font-weight: bold;
          border: 10px solid yellow;
          border-radius: 20px;
      }
      </style>
      <div class="extreme-alert">
          🚨🚨 SYSTEM CRITICAL: EMERGENCY 🚨🚨
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
    
    if 'last_anomaly_time_ml' not in st.session_state:
      st.session_state.last_anomaly_time_ml = 0
    
    for i in range(st.session_state.current_idx, len(display_df)):
      st.session_state.current_idx = i
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

      if is_anomaly:
        # 이상 징후가 발견된 시점의 시간을 세션에 박제
        st.session_state.last_anomaly_time_ml = time.time()
      # 마지막 이상 발생 후 10초(원하는 만큼 조절) 동안은 경고 유지
      is_maintaining_alert = (time.time() - st.session_state.last_anomaly_time_ml) < 10
    
      # 3. 화면 표시
      with status_box.container():
        if is_maintaining_alert and not st.session_state.mute_alert:
          trigger_emergency_alert()  # 화면 흔들기/빨간색 유지
          
          # 남은 유지 시간 표시 (선택 사항)
          rem = int(10 - (time.time() - st.session_state.last_anomaly_time_ml))
          if rem > 0:
            st.toast(f"경고가 {rem}초간 더 유지됩니다.")
            
        # 2. 주의 단계 (점수가 낮아지는 중)
        elif score < WARNING_THRESHOLD and not is_anomaly:
          st.warning(f"⚠️ [주의] 이상 전조 증상 포착! (점수: {score:.4f})")
            
        # 3. 정상
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
        fig.add_hrect(y0=DANGER_LINE, y1=WARNING_THRESHOLD, 
                      fillcolor="yellow", opacity=0.3, line_width=0,
                      annotation_text="Warning Zone (Pre-detection)", 
                      annotation_position="top left")
    
        # [핵심] 빨간색 '위험' 영역 표시
        fig.add_hrect(y0=-0.5, y1=DANGER_LINE,  # min() 대신 -0.5 같은 고정값 사용
                  fillcolor="red", opacity=0.2, line_width=0,
                  annotation_text="Danger Zone", 
                  annotation_position="bottom left")
    
        # 기준선(Line) 추가
        fig.add_hline(y=DANGER_LINE, line_dash="dash", line_color="red")
        fig.add_hline(y=WARNING_THRESHOLD, line_dash="dot", line_color="orange")
        
        # y축 범위를 점수에 맞게 조정
        fig.update_layout(yaxis=dict(range=[-0.5, 0.5], # 고정값 설정 (데이터 특성에 맞게 조절 가능)
                    fixedrange=True,    # 사용자가 마우스로 드래그해서 축이 변하는 것도 방지
                    autorange=False        # 데이터에 따라 축이 변하는 기능 강제 종료
                ),
                height=400,
                margin=dict(t=50, b=20),
                transition_duration=50 # 선이 뚝뚝 끊기지 않고 부드럽게 흐르도록 설정
            )
        st.plotly_chart(fig, use_container_width=True, key=f"pre_det_chart_{i}")

      time.sleep(0.1)
  
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
                  trigger_emergency_alert()  # 빨간색 깜빡이 효과 유지
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

            if auto_dl_analyze:
              current_time = time.time()
              # 10초 주기로 SHAP 분석 서버 호출
              if current_time - st.session_state.last_shap_time > 10:
                # SHAP 분석에는 최신 100개의 시계열 데이터(Window)가 필요함
                # df_input이 실시간으로 쌓이고 있다면 tail(100)을 사용
                if len(df_input) >= 100:
                  target_window = df_input[new_column_names].tail(100).values.tolist()
                  
                  shap_payload = {
                      "machine_name": selected_machine,
                      "window": target_window
                  }
                  
                  try:
                    # SHAP 서버는 분석 시간이 걸리므로 timeout을 넉넉히 줌
                    shap_resp = requests.post(SHAP_URL, json=shap_payload, timeout=15)
                    
                    if shap_resp.status_code == 200:
                      res_data = shap_resp.json()
                      if res_data.get("status") == "success":
                        # 1. 중요도 데이터 가공
                        importance_dict = res_data["importance"]
                        imp_df_new = pd.DataFrame(list(importance_dict.items()), 
                                                  columns=['Feature', 'Importance'])
                        # 2. 절대값 기준으로 영향력 정렬
                        imp_df_new['Abs_Importance'] = imp_df_new['Importance'].abs()
                        st.session_state['dl_importance'] = imp_df_new.sort_values(by='Abs_Importance', ascending=False)
                        
                        st.session_state.last_shap_time = current_time
                        # st.toast("💡 원인 분석 결과가 갱신되었습니다.") 
                  except Exception as e:
                      # SHAP 서버 에러가 탐지 루프를 멈추지 않도록 예외 처리
                      pass
            
        else:
          # 데이터 수집 단계 (WIN_SIZE 100개 채우는 중) 
          status_box.info(f"⏳ 서버 데이터 축적 중... ({res['progress']})")
      
      else:
        st.error(f"서버 오류: {response.status_code}")
        break
            
    except Exception as e:
      st.error(f"연결 실패: {e}")
      break

# ------------------ [여기서부터 SHAP 로직 추가] ------------------
    if auto_dl_analyze:
      current_time = time.time()
      # 10초 주기로 SHAP 분석 서버 호출
      if current_time - st.session_state.last_shap_time > 10:
        # SHAP 분석에는 최신 100개의 시계열 데이터(Window)가 필요함
        # df_input이 실시간으로 쌓이고 있다면 tail(100)을 사용
        if len(df_input) >= 100:
          target_window = df_input[new_column_names].tail(100).values.tolist()
          
          shap_payload = {
              "machine_name": selected_machine,
              "window": target_window
          }
          
          try:
            # SHAP 서버는 분석 시간이 걸리므로 timeout을 넉넉히 줌
            shap_resp = requests.post(SHAP_URL, json=shap_payload, timeout=15)
            
            if shap_resp.status_code == 200:
              res_data = shap_resp.json()
              if res_data.get("status") == "success":
                # 1. 중요도 데이터 가공
                importance_dict = res_data["importance"]
                imp_df_new = pd.DataFrame(list(importance_dict.items()), 
                                          columns=['Feature', 'Importance'])
                # 2. 절대값 기준으로 영향력 정렬
                imp_df_new['Abs_Importance'] = imp_df_new['Importance'].abs()
                st.session_state['dl_importance'] = imp_df_new.sort_values(by='Abs_Importance', ascending=False)
                
                st.session_state.last_shap_time = current_time
                # st.toast("💡 원인 분석 결과가 갱신되었습니다.") 
          except Exception as e:
              # SHAP 서버 에러가 탐지 루프를 멈추지 않도록 예외 처리
              pass
time.sleep(0.2) # test_client.py의 전송 속도와 맞춤

with st.expander('Data'):
  st.write('**Raw Data**')
  df_input

# --- 공통 시각화 (모델이 선정한 주요 지표 Top 5 시각화) ---
with st.expander('🔍 Top 5 Influential Features Detail View'):
    st.write("모델이 분석한 이상 징후 기여도 상위 5개 지표의 변화 추이입니다.")
    
    viz_cols = []
    if "DL" in model_type:
        # 루프에서 10초마다 세션에 저장해주는 따끈따끈한 데이터를 가져옴
        if 'dl_importance' in st.session_state:
            viz_cols = st.session_state['dl_importance'].head(5)['Feature'].tolist()
        
    # 추출된 컬럼 시각화 (기존 px.line 로직 그대로 사용)
    if viz_cols:
        for idx, col in enumerate(viz_cols):
            fig = px.line(df_input.tail(100), y=col, title=f"Top {idx+1} 원인 지표: {col}")
            st.plotly_chart(fig, use_container_width=True)
