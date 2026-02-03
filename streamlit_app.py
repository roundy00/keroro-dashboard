import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import numpy as np
import shap

# =====================================================================
# 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="실시간 서버 모니터링",
    page_icon="🚀",
    layout="wide"
)

st.title('🤖 서버 실시간 이상 탐지 대시보드 🤖')
st.info('이 앱은 머신러닝과 딥러닝 모델을 활용하여 실시간으로 서버 상태를 모니터링합니다')

# =====================================================================
# 전역 설정
# =====================================================================
machine_num = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
               '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
               '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11']

# API 서버 주소
SHAP_API_URL = "https://unbarreled-uncrusted-juliana.ngrok-free.dev/analyze"
ANOMALY_API_URL = "https://unbarreled-uncrusted-juliana.ngrok-free.dev/predict"

# 피처 이름
new_column_names = [
    'cpu_r', 'load_1', 'load_5', 'load_15', 'mem_shmem', 'mem_u', 'mem_u_e', 'total_mem',
    'disk_q', 'disk_r', 'disk_rb', 'disk_svc', 'disk_u', 'disk_w', 'disk_wa', 'disk_wb',
    'si', 'so', 'eth1_fi', 'eth1_fo', 'eth1_pi', 'eth1_po', 'tcp_tw', 'tcp_use',
    'active_opens', 'curr_estab', 'in_errs', 'in_segs', 'listen_overflows', 'out_rsts',
    'out_segs', 'passive_opens', 'retransegs', 'tcp_timeouts', 'udp_in_dg', 'udp_out_dg',
    'udp_rcv_buf_errs', 'udp_snd_buf_errs'
]

# =====================================================================
# 세션 상태 초기화
# =====================================================================
def init_session_state():
    """세션 상태 변수들을 초기화"""
    if 'workflow_stage' not in st.session_state:
        st.session_state.workflow_stage = 'SELECT_MODEL'  # SELECT_MODEL, SHAP_ANALYSIS, REALTIME_MONITORING
    
    if 'shap_completed' not in st.session_state:
        st.session_state.shap_completed = False
    
    if 'shap_results' not in st.session_state:
        st.session_state.shap_results = None
    
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    
    if 'anomaly_scores' not in st.session_state:
        st.session_state.anomaly_scores = []
    
    if 'last_anomaly_time' not in st.session_state:
        st.session_state.last_anomaly_time = 0
    
    if 'mute_alert' not in st.session_state:
        st.session_state.mute_alert = False

init_session_state()

# =====================================================================
# 사이드바 설정
# =====================================================================
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    # 머신 선택
    selected_machine = st.selectbox('대상 머신 선택', [f'machine-{i}' for i in machine_num])
    
    # 모델 선택
    model_type = st.radio(
        '분석 모델 종류',
        ["ML (IsolationForest)", "DL (Anomaly Transformer)"],
        key='model_type_radio'
    )
    
    st.divider()
    
    # 긴급 제어
    st.header("🚨 긴급 제어")
    
    if st.button("🔄 전체 초기화", use_container_width=True):
        # 모든 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.button("🔇 경보 음소거", use_container_width=True):
        st.session_state.mute_alert = True
        st.success("경보가 음소거되었습니다")
    
    st.divider()
    
    # 시작 인덱스 설정 (딥러닝 모델에서만 사용)
    if 'DL' in model_type:
        start_point = st.number_input(
            '시작 인덱스 설정',
            min_value=0,
            max_value=10000,
            value=0,
            step=100
        )
    else:
        start_point = 0

# =====================================================================
# 데이터 로드
# =====================================================================
@st.cache_data
def load_data(machine_name):
    """데이터 로드 및 전처리"""
    df_train = pd.read_csv(
        f'https://raw.githubusercontent.com/roundy00/keroro-machinelearning/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{machine_name}/{machine_name}_train.csv'
    )
    df_test = pd.read_csv(
        f'https://raw.githubusercontent.com/roundy00/keroro-dashboard/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{machine_name}/{machine_name}_test.csv'
    )
    
    # 컬럼 이름 변경
    rename_dict = {f'col_{i}': new_column_names[i] for i in range(len(new_column_names))}
    df_train.rename(columns=rename_dict, inplace=True)
    df_test.rename(columns=rename_dict, inplace=True)
    
    return df_train, df_test

# 데이터 로드
with st.spinner('데이터 로딩 중...'):
    df_train, df_test = load_data(selected_machine)

# 시작 인덱스 적용
df_test_display = df_test.iloc[start_point:].reset_index(drop=True)

# =====================================================================
# 현재 상태 표시
# =====================================================================
status_cols = st.columns([2, 2, 1])
with status_cols[0]:
    st.metric("머신", selected_machine)
with status_cols[1]:
    st.metric("모델", model_type)
with status_cols[2]:
    stage_emoji = {
        'SELECT_MODEL': '1️⃣',
        'SHAP_ANALYSIS': '2️⃣',
        'REALTIME_MONITORING': '3️⃣'
    }
    st.metric("단계", stage_emoji.get(st.session_state.workflow_stage, '❓'))

st.divider()

# =====================================================================
# 유틸리티 함수
# =====================================================================
def trigger_emergency_alert():
    """긴급 경보 UI"""
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

# =====================================================================
# 머신러닝 모델 (Isolation Forest) - 2단계 워크플로우
# =====================================================================
if model_type == "ML (IsolationForest)":
    
    # 모델 학습 (최초 1회)
    if 'ml_model_trained' not in st.session_state:
        X_train = df_train.drop(columns=['timestamp'], axis=1)
        model = IsolationForest(contamination=0.01, random_state=42)
        
        with st.spinner('모델 학습 중...'):
            model.fit(X_train)
        
        st.session_state.ml_model = model
        st.session_state.ml_model_trained = True
    else:
        model = st.session_state.ml_model
    
    # 파라미터 설정
    DANGER_LINE = 0.0
    WARNING_THRESHOLD = 0.05
    
    # 세션 변수 초기화
    if 'ml_workflow_stage' not in st.session_state:
        st.session_state.ml_workflow_stage = 'SHAP_FIRST'  # SHAP_FIRST, MONITORING
    
    if 'ml_scores_history' not in st.session_state:
        st.session_state.ml_scores_history = []
    
    if 'last_anomaly_time_ml' not in st.session_state:
        st.session_state.last_anomaly_time_ml = 0
    
    if 'ml_shap_completed' not in st.session_state:
        st.session_state.ml_shap_completed = False
    
    # ========== 단계 1: SHAP 분석 먼저 ==========
    if st.session_state.ml_workflow_stage == 'SHAP_FIRST':
        st.header("🤖 머신러닝 기반 이상 탐지 (Isolation Forest)")
        st.success("✅ 모델 학습 완료!")
        
        st.divider()
        st.subheader("1️⃣ 단계 1: SHAP 원인 분석")
        
        if not st.session_state.ml_shap_completed:
            st.info("📊 먼저 SHAP 분석을 통해 주요 이상 원인 지표를 파악합니다")
            
            if st.button("🚀 SHAP 분석 실행", use_container_width=True, type="primary"):
                with st.spinner("SHAP 분석 중..."):
                    explainer = shap.TreeExplainer(model)
                    X_sample = df_test_display[new_column_names].sample(min(100, len(df_test_display)))
                    shap_values = explainer.shap_values(X_sample)
                    
                    importance = np.abs(shap_values).mean(axis=0)
                    analysis_df = pd.DataFrame({
                        'Feature': new_column_names,
                        'Importance': importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    st.session_state.ml_shap_results = analysis_df
                    st.session_state.ml_shap_completed = True
                    st.success("✅ SHAP 분석 완료!")
                    time.sleep(1)
                    st.rerun()
        
        else:
            # SHAP 결과 표시
            st.success("✅ SHAP 분석이 완료되었습니다!")
            
            if st.session_state.ml_shap_results is not None:
                top_features = st.session_state.ml_shap_results.head(10)
                
                fig = px.bar(
                    top_features[::-1],
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="주요 이상 원인 지표 TOP 10",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 데이터 표시
                with st.expander("📋 전체 분석 결과 보기"):
                    st.dataframe(
                        st.session_state.ml_shap_results,
                        use_container_width=True,
                        hide_index=True
                    )
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 SHAP 재분석", use_container_width=True):
                    st.session_state.ml_shap_completed = False
                    st.rerun()
            
            with col2:
                if st.button("▶️ 실시간 모니터링 시작", use_container_width=True, type="primary"):
                    st.session_state.ml_workflow_stage = 'MONITORING'
                    st.session_state.current_idx = 0
                    st.session_state.ml_scores_history = []
                    st.rerun()
    
    # ========== 단계 2: 실시간 모니터링 ==========
    elif st.session_state.ml_workflow_stage == 'MONITORING':
        st.header("🤖 머신러닝 기반 이상 탐지 (Isolation Forest)")
        
        # SHAP 결과 요약 표시
        with st.expander("📊 SHAP 분석 결과 요약"):
            if st.session_state.ml_shap_results is not None:
                top_3 = st.session_state.ml_shap_results.head(3)
                cols = st.columns(3)
                for idx, row in enumerate(top_3.itertuples()):
                    with cols[idx]:
                        st.metric(
                            f"#{idx+1} {row.Feature}",
                            f"{row.Importance:.4f}"
                        )
        
        st.divider()
        st.subheader("2️⃣ 단계 2: 실시간 이상 탐지 모니터링")
        
        # 실시간 분석 UI
        status_box = st.empty()
        alert_box = st.empty()
        chart_box = st.empty()
        metrics_box = st.empty()
        
        # 실시간 분석 루프
        for i in range(st.session_state.current_idx, len(df_test_display)):
            st.session_state.current_idx = i
            current_row = df_test_display.iloc[i:i+1][new_column_names]
            
            # 이상 점수 계산
            score = model.decision_function(current_row)[0]
            st.session_state.ml_scores_history.append({
                'step': i,
                'score': score,
                'is_anomaly': score < DANGER_LINE
            })
            
            # 이상 탐지
            is_anomaly = score < DANGER_LINE
            is_warning = (score < WARNING_THRESHOLD) and (not is_anomaly)
            
            if is_anomaly:
                st.session_state.last_anomaly_time_ml = time.time()
            
            is_maintaining_alert = (time.time() - st.session_state.last_anomaly_time_ml) < 10
            
            # 경고 표시
            now = time.time()
            if now - st.session_state.last_anomaly_time_ml < 10:
                with alert_box.container():
                    if not st.session_state.mute_alert:
                        trigger_emergency_alert()
                    remaining = int(10 - (now - st.session_state.last_anomaly_time_ml))
                    st.toast(f"🚨 이상 감지! ({remaining}초 유지)")
            else:
                alert_box.empty()
            
            # 상태 표시
            with status_box.container():
                if is_anomaly:
                    st.error(f"🚨 **이상 발생!** 점수: {score:.4f} | 임계치: {DANGER_LINE:.4f}")
                elif is_warning:
                    st.warning(f"⚠️ **주의** 이상 전조 증상 포착! (점수: {score:.4f})")
                else:
                    st.success(f"✅ **정상 상태** 점수: {score:.4f}")
            
            # 차트 업데이트
            with chart_box.container():
                if len(st.session_state.ml_scores_history) > 0:
                    # 최근 100개 데이터
                    recent_data = st.session_state.ml_scores_history[-100:]
                    df_chart = pd.DataFrame(recent_data)
                    
                    # 메인 차트
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    
                    # 정상 구간
                    normal_data = df_chart[~df_chart['is_anomaly']]
                    fig.add_trace(go.Scatter(
                        x=normal_data['step'],
                        y=normal_data['score'],
                        mode='lines+markers',
                        name='정상',
                        line=dict(color='green', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # 이상 구간
                    anomaly_data = df_chart[df_chart['is_anomaly']]
                    if len(anomaly_data) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomaly_data['step'],
                            y=anomaly_data['score'],
                            mode='markers',
                            name='이상',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    # 임계치 영역
                    fig.add_hrect(
                        y0=DANGER_LINE, y1=WARNING_THRESHOLD,
                        fillcolor="yellow", opacity=0.3, line_width=0,
                        annotation_text="주의 구간",
                        annotation_position="top left"
                    )
                    fig.add_hrect(
                        y0=-0.5, y1=DANGER_LINE,
                        fillcolor="red", opacity=0.2, line_width=0,
                        annotation_text="위험 구간",
                        annotation_position="bottom left"
                    )
                    
                    fig.add_hline(
                        y=DANGER_LINE,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"위험 임계치: {DANGER_LINE}",
                        annotation_position="right"
                    )
                    
                    fig.update_layout(
                        title=f"{selected_machine} 실시간 이상 탐지 점수 (최근 100개)",
                        xaxis_title="Time Step",
                        yaxis_title="Anomaly Score",
                        yaxis=dict(range=[-0.5, 0.5], fixedrange=True),
                        height=450,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"ml_chart_{i}")
            
            # 메트릭 표시
            with metrics_box.container():
                cols = st.columns(4)
                with cols[0]:
                    st.metric("현재 Step", i)
                with cols[1]:
                    st.metric("총 데이터 수", len(st.session_state.ml_scores_history))
                with cols[2]:
                    anomaly_count = sum(1 for d in st.session_state.ml_scores_history if d['is_anomaly'])
                    st.metric("이상 감지 횟수", anomaly_count)
                with cols[3]:
                    if len(st.session_state.ml_scores_history) > 0:
                        anomaly_rate = anomaly_count / len(st.session_state.ml_scores_history) * 100
                        st.metric("이상 발생률", f"{anomaly_rate:.2f}%")
            
            time.sleep(0.1)

# =====================================================================
# 딥러닝 모델 (Anomaly Transformer) - 3단계 워크플로우
# =====================================================================
elif model_type == "DL (Anomaly Transformer)":
    # 1. 독립적인 레이아웃 구성을 위한 탭 생성
    tab1, tab2 = st.tabs(["📊 실시간 모니터링", "🔍 SHAP 원인 분석"])

    # ---------------------------------------------------------
    # TAB 1: 실시간 모니터링 (데이터 1개부터 즉시 가능)
    # ---------------------------------------------------------
    with tab1:
        st.subheader("3️⃣ 실시간 이상 탐지 모니터링")
        
        # 시작/중지 버튼 제어
        if not st.session_state.monitoring_active:
            if st.button("▶️ 모니터링 시작", use_container_width=True, type="primary"):
                st.session_state.monitoring_active = True
                st.rerun()
        else:
            if st.button("🛑 모니터링 중지", use_container_width=True):
                st.session_state.monitoring_active = False
                st.rerun()

        if st.session_state.monitoring_active:
            status_box = st.empty()
            alert_box = st.empty()
            chart_box = st.empty()
            metrics_box = st.empty()

            # 모니터링 루프 (기존 로직 유지하되 세션 상태 업데이트 방식 최적화)
            for i in range(st.session_state.current_idx, len(df_test_display)):
                if not st.session_state.monitoring_active: break # 중지 버튼 대응
                
                st.session_state.current_idx = i
                current_row = df_test_display.iloc[i][new_column_names].values.tolist()

                try:
                    response = requests.post(ANOMALY_API_URL, json={"values": current_row}, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('status') == "ready":
                            # 결과 파싱 및 점수 저장
                            score = result.get('score', 0.0)
                            threshold = result.get('threshold', 0.5)
                            is_anomaly = result.get('is_anomaly', False)
                            st.session_state.anomaly_scores.append({'step': i, 'score': score, 'is_anomaly': is_anomaly})

                            # [UI 업데이트] 상태 메시지 및 차트 출력
                            with status_box:
                                if is_anomaly: st.error(f"🚨 이상 발생! 점수: {score:.6f}")
                                else: st.success(f"✅ 정상 상태 점수: {score:.6f}")
                            
                            # 1. 경고 및 알림 로직 (최근 30초 내 이상 발생 시)
                            now = time.time()
                            if is_anomaly:
                                st.session_state.last_anomaly_time = now
                                st.toast(f"🚨 이상 감지! (Step: {i})")
                            
                            # 2. 실시간 차트 업데이트 (최근 100개 데이터)
                            with chart_box:
                                if len(st.session_state.anomaly_scores) > 0:
                                    recent_data = st.session_state.anomaly_scores[-100:]
                                    df_chart = pd.DataFrame(recent_data)
                                    
                                    fig = go.Figure()
                                    
                                    # 정상 데이터 (초록색 선)
                                    normal_data = df_chart[~df_chart['is_anomaly']]
                                    fig.add_trace(go.Scatter(
                                        x=df_chart['step'], y=df_chart['score'],
                                        mode='lines', name='Score',
                                        line=dict(color='#2ecc71', width=2)
                                    ))
                                    
                                    # 이상 지점 (빨간색 X 표시)
                                    anomaly_pts = df_chart[df_chart['is_anomaly']]
                                    if not anomaly_pts.empty:
                                        fig.add_trace(go.Scatter(
                                            x=anomaly_pts['step'], y=anomaly_pts['score'],
                                            mode='markers', name='Anomaly',
                                            marker=dict(color='#e74c3c', size=10, symbol='x')
                                        ))
                                    
                                    # 임계치 점선
                                    fig.add_hline(y=threshold, line_dash="dash", line_color="#c0392b", 
                                                  annotation_text=f"Threshold: {threshold:.4f}")
                                    
                                    fig.update_layout(
                                        title=f"실시간 이상 탐지 스코어 (최근 100 step)",
                                        xaxis_title="Time Step", yaxis_title="Score",
                                        height=400, margin=dict(l=10, r=10, t=40, b=10),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig, use_container_width=True, key=f"dl_chart_{i}")

                            # 3. 실시간 메트릭 카드 업데이트
                            with metrics_box:
                                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                                anomaly_count = sum(1 for d in st.session_state.anomaly_scores if d['is_anomaly'])
                                total_count = len(st.session_state.anomaly_scores)
                                anomaly_rate = (anomaly_count / total_count * 100) if total_count > 0 else 0
                                
                                m_col1.metric("현재 Step", i)
                                m_col2.metric("현재 스코어", f"{score:.4f}", delta=f"{score-threshold:.4f}" if is_anomaly else None, delta_color="inverse")
                                m_col3.metric("이상 감지 횟수", f"{anomaly_count}회")
                                m_col4.metric("이상 발생률", f"{anomaly_rate:.2f}%")
                                
                        else:
                            status_box.info(f"⏳ 서버 데이터 축적 중... ({result.get('progress')})")
                except Exception as e:
                    st.error(f"⚠️ 통신 오류: {e}")
                time.sleep(0.1)

    # ---------------------------------------------------------
    # TAB 2: SHAP 분석 (100개 데이터 의존성 해결)
    # ---------------------------------------------------------
    with tab2:
        st.subheader("🔍 인공지능 판단 근거 분석 (SHAP)")
        
        current_data_count = len(df_test_display)
        WINDOW_SIZE = 100

        if current_data_count < WINDOW_SIZE:
            st.warning(f"⚠️ SHAP 분석을 위해 최소 {WINDOW_SIZE}개의 데이터가 필요합니다. (현재: {current_data_count}개)")
            st.info("💡 실시간 모니터링을 실행하여 데이터를 축적하거나 사이드바에서 시작 인덱스를 조정하세요.")
            st.button("🚀 SHAP 분석 실행", disabled=True, use_container_width=True)
        else:
            # 분석 결과가 아직 없을 때
            if not st.session_state.get('shap_completed', False):
                st.info("📊 현재 시점으로부터 과거 100개 데이터를 분석하여 이상 징후에 기여하는 주요 지표를 도출합니다.")
                if st.button("🚀 SHAP 분석 실행", use_container_width=True, type="primary"):
                    with st.spinner("SHAP 서버에서 분석 중... (약 30~60초 소요)"):
                        try:
                            # 1. 윈도우 데이터 준비
                            raw_data = df_test_display[new_column_names].values
                            window_data = raw_data[-WINDOW_SIZE:].tolist()
                            
                            # 2. SHAP API 호출
                            response = requests.post(
                                SHAP_API_URL, timeout = 120)
                            st.write("SHAP response status:", response.status_code)
                            st.write("SHAP response text:", response.text)
                            
                            if response.status_code == 200:
                                result = response.json()
                                importance_data = result.get('importance', {})
                                if result.get("status") != "success":
                                    st.error(f"SHAP 서버 실패: {result}")
                                    st.stop()
                                
                                importance_data = result.get("importance")
                                if not importance_data:
                                    st.error("SHAP 결과 importance가 비어 있습니다.")
                                    st.stop()
                                    
                                if importance_data:
                                    # 3. 데이터프레임 변환 및 정렬
                                    imp_df = pd.DataFrame([
                                        {'Feature': k, 'Importance': v}
                                        for k, v in importance_data.items()
                                    ])
                                    imp_df['Abs_Importance'] = imp_df['Importance'].abs()
                                    imp_df = imp_df.sort_values(by='Abs_Importance', ascending=False)
                                    
                                    # 4. 세션 상태 저장
                                    st.session_state.shap_results = imp_df
                                    st.session_state.shap_completed = True
                                    st.success("✅ SHAP 분석 완료!")
                                    st.rerun()
                                else:
                                    st.error("❌ 분석 결과가 비어있습니다.")
                            else:
                                st.error(f"❌ 서버 오류: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ 분석 중 오류 발생: {str(e)}")

            # 분석 결과가 있을 때 시각화 출력
            else:
                if st.session_state.shap_results is not None:
                    st.success("📊 SHAP 분석 결과: 주요 기여 지표 TOP 10")
                    
                    # 시각화 1: 막대 그래프 (Plotly)
                    top_features = st.session_state.shap_results.head(10)
                    fig = px.bar(
                        top_features[::-1], # 상위 항목이 위로 오게 역순
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='RdYlGn_r',
                        labels={'Importance': '영향력 (Shapley Value)'}
                    )
                    fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 시각화 2: 상세 데이터 테이블
                    with st.expander("📋 전체 피처 영향력 데이터 보기"):
                        st.dataframe(
                            st.session_state.shap_results,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # 다시 분석하기 버튼
                    if st.button("🔄 새로운 데이터로 재분석", use_container_width=True):
                        st.session_state.shap_completed = False
                        st.session_state.shap_results = None
                        st.rerun()

# =====================================================================
# 하단 데이터 표시
# =====================================================================
st.divider()
with st.expander("📋 원본 데이터 보기"):
    st.dataframe(df_test_display, use_container_width=True)
