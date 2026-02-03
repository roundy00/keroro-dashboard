import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# =====================================================================
# 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="🚀 Keroro 서버 모니터링 시스템",
    page_icon="🚀",
    layout="wide"
)

# 커스텀 CSS로 UI 개선
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-normal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stTab {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 Keroro 서버 실시간 이상 탐지 시스템</h1>', unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 전역 설정
# =====================================================================
machine_num = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
               '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
               '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11']

FEATURE_NAMES = [
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
    defaults = {
        'base_url': 'https://unbarreled-uncrusted-juliana.ngrok-free.dev',
        'model_type': 'DL',
        'workflow_stage': 'SELECT_MODEL',
        'shap_completed': False,
        'shap_results': None,
        'monitoring_active': False,
        'stream_running': False,
        'stream_i': 0,
        'stream_target': None,
        'history': [],
        'last_result': None,
        'alert_until': 0.0,
        'alert_msg': '',
        'anomaly_scores': [],
        'last_anomaly_time': 0,
        'mute_alert': False,
        'FILE_PATH': '',
        'START_INDEX': 15800,
        'SLEEP_SEC': 0.2,
        'N_PER_RERUN': 5,
        'npy_data_loaded': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =====================================================================
# 사이드바 설정
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ 시스템 설정")
    
    # 서버 URL 설정
    base_url = st.text_input(
        "🔗 FastAPI 서버 URL",
        value=st.session_state.base_url,
        help="예: https://xxxx.ngrok-free.dev"
    ).strip().rstrip("/")
    st.session_state.base_url = base_url
    
    # Health Check
    if st.button("💓 서버 상태 확인", use_container_width=True):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                st.success(f"✅ 서버 정상: {r.json()}")
            else:
                st.error(f"❌ HTTP {r.status_code}")
        except Exception as e:
            st.error(f"❌ 연결 실패: {e}")
    
    st.divider()
    
    # 모델 선택
    model_type = st.radio(
        '🤖 분석 모델 선택',
        ["ML (IsolationForest)", "DL (Anomaly Transformer)"],
        index=1 if st.session_state.model_type == 'DL' else 0
    )
    st.session_state.model_type = 'DL' if 'DL' in model_type else 'ML'
    
    st.divider()
    
    # 머신 선택
    selected_machine = st.selectbox(
        '🖥️ 대상 머신 선택',
        [f'machine-{i}' for i in machine_num],
        index=0
    )
    
    st.divider()
    
    # DL 모델용 추가 설정
    if st.session_state.model_type == 'DL':
        st.markdown("#### 📊 DL 모델 설정")
        
        # GitHub에서 자동 로드
        st.info(f"📂 GitHub에서 {selected_machine}.npy 로드")
        
        # 로드 버튼
        if st.button("🔄 데이터 로드", use_container_width=True):
            with st.spinner("GitHub에서 데이터 다운로드 중..."):
                data, temp_path = load_npy_from_github(selected_machine)
                if data is not None:
                    st.session_state.FILE_PATH = temp_path
                    st.session_state.npy_data_loaded = True
                    st.success(f"✅ 로드 완료: {data.shape}")
                else:
                    st.error("데이터 로드 실패")
        
        start_index = st.number_input(
            "시작 인덱스",
            min_value=0,
            value=st.session_state.START_INDEX,
            step=100
        )
        st.session_state.START_INDEX = int(start_index)
        
        sleep_sec = st.number_input(
            "갱신 주기 (초)",
            min_value=0.01,
            value=st.session_state.SLEEP_SEC,
            step=0.01
        )
        st.session_state.SLEEP_SEC = float(sleep_sec)
        
        n_per_rerun = st.number_input(
            "배치 크기",
            min_value=1,
            value=st.session_state.N_PER_RERUN,
            step=1
        )
        st.session_state.N_PER_RERUN = int(n_per_rerun)
    
    st.divider()
    
    # 긴급 제어
    st.markdown("### 🚨 긴급 제어")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 초기화", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🔇 음소거", use_container_width=True):
            st.session_state.mute_alert = True
            st.success("음소거됨")

# =====================================================================
# 데이터 로드 함수
# =====================================================================
@st.cache_data
def load_data(machine_name):
    """ML 모드용 데이터 로드"""
    df_train = pd.read_csv(
        f'https://raw.githubusercontent.com/roundy00/keroro-machinelearning/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{machine_name}/{machine_name}_train.csv'
    )
    df_test = pd.read_csv(
        f'https://raw.githubusercontent.com/roundy00/keroro-dashboard/refs/heads/master/Server-Machine-Dataset-main/processed_csv/{machine_name}/{machine_name}_test.csv'
    )
    
    rename_dict = {f'col_{i}': FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES))}
    df_train.rename(columns=rename_dict, inplace=True)
    df_test.rename(columns=rename_dict, inplace=True)
    
    return df_train, df_test

@st.cache_data
def load_npy_from_github(machine_name):
    """GitHub에서 NPY 파일 로드"""
    # GitHub raw URL
    url = f'https://github.com/roundy00/keroro-dashboard/raw/master/{machine_name}.npy'
    
    try:
        # 임시 파일로 다운로드
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 임시 파일에 저장
        temp_path = f'/tmp/{machine_name}.npy'
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # numpy 배열로 로드
        data = np.load(temp_path)
        return data, temp_path
    
    except Exception as e:
        st.error(f"GitHub에서 파일 로드 실패: {e}")
        return None, None

# =====================================================================
# 유틸리티 함수
# =====================================================================
def trigger_emergency_alert():
    """긴급 경보 UI"""
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff0000 0%, #ff6b6b 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            border-radius: 15px;
            border: 5px solid #ffeb3b;
            box-shadow: 0 0 30px rgba(255,0,0,0.5);
            animation: pulse 0.5s infinite;
        ">
            🚨 시스템 경고: 이상 감지됨! 🚨
        </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 메인 탭 구성
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "📈 실시간 모니터링",
    "🔍 SHAP 분석",
    "📊 시스템 정보"
])

# =====================================================================
# TAB 1: 실시간 모니터링
# =====================================================================
with tab1:
    st.markdown("## 📈 실시간 이상 탐지 모니터링")
    
    # 현재 상태 표시
    cols = st.columns(4)
    with cols[0]:
        st.markdown('<div class="metric-card">🖥️<br>머신<br><h3>{}</h3></div>'.format(selected_machine), unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="metric-card">🤖<br>모델<br><h3>{}</h3></div>'.format(st.session_state.model_type), unsafe_allow_html=True)
    with cols[2]:
        stage_name = {
            'SELECT_MODEL': '모델 선택',
            'SHAP_ANALYSIS': 'SHAP 분석',
            'REALTIME_MONITORING': '실시간 모니터링'
        }.get(st.session_state.workflow_stage, '알 수 없음')
        st.markdown('<div class="metric-card">📍<br>단계<br><h3>{}</h3></div>'.format(stage_name), unsafe_allow_html=True)
    with cols[3]:
        alert_count = len([x for x in st.session_state.history if x.get('is_anomaly', False)])
        st.markdown('<div class="metric-card">⚠️<br>이상 횟수<br><h3>{}</h3></div>'.format(alert_count), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # DL 모드
    if st.session_state.model_type == 'DL':
        st.markdown("### 🎯 딥러닝 기반 실시간 예측")
        
        # 컨트롤 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            start_btn = st.button("▶️ 모니터링 시작", use_container_width=True, type="primary")
        with col2:
            stop_btn = st.button("⏹ 중지", use_container_width=True)
        with col3:
            reset_btn = st.button("🧹 데이터 초기화", use_container_width=True)
        
        if reset_btn:
            st.session_state.stream_running = False
            st.session_state.stream_i = 0
            st.session_state.stream_target = None
            st.session_state.history = []
            st.session_state.last_result = None
            st.session_state.alert_until = 0.0
            st.session_state.alert_msg = ""
            st.success("✅ 초기화 완료")
        
        if start_btn:
            # 데이터가 로드되지 않았으면 자동으로 GitHub에서 로드
            if not st.session_state.npy_data_loaded or not st.session_state.FILE_PATH:
                with st.spinner("GitHub에서 데이터 다운로드 중..."):
                    data, temp_path = load_npy_from_github(selected_machine)
                    if data is not None:
                        st.session_state.FILE_PATH = temp_path
                        st.session_state.npy_data_loaded = True
                        all_data = data
                    else:
                        st.error("❌ GitHub에서 데이터를 로드할 수 없습니다.")
                        all_data = None
            else:
                # 이미 로드된 파일 사용
                if os.path.exists(st.session_state.FILE_PATH):
                    all_data = np.load(st.session_state.FILE_PATH)
                else:
                    st.error(f"❌ 파일을 찾을 수 없습니다: {st.session_state.FILE_PATH}")
                    all_data = None
            
            if all_data is not None:
                target_data = all_data[st.session_state.START_INDEX:]
                st.session_state.stream_target = target_data
                st.session_state.stream_i = 0
                st.session_state.stream_running = True
                st.success(f"📂 데이터 로드 완료: {all_data.shape} | 시작: {st.session_state.START_INDEX}")
        
        if stop_btn:
            st.session_state.stream_running = False
            st.info("⏸ 모니터링 중지")
        
        # Placeholders
        alert_placeholder = st.empty()
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # 차트 그리기
        df_h = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame(
            columns=["index", "score", "threshold", "is_anomaly", "over_threshold"]
        )
        
        fig = go.Figure()
        
        if len(df_h) > 0:
            # Score line
            fig.add_trace(go.Scatter(
                x=df_h["index"],
                y=df_h["score"],
                mode="lines+markers",
                name="점수",
                line=dict(color='#667eea', width=2),
                marker=dict(size=6)
            ))
            
            # Anomaly markers
            if "is_anomaly" in df_h.columns:
                an = df_h[df_h["is_anomaly"] == True]
                if len(an) > 0:
                    fig.add_trace(go.Scatter(
                        x=an["index"],
                        y=an["score"],
                        mode="markers",
                        name="이상",
                        marker=dict(symbol="x", size=12, color='red')
                    ))
            
            # Threshold line
            if "threshold" in df_h.columns and df_h["threshold"].notna().any():
                thr_last = float(df_h["threshold"].iloc[-1])
                fig.add_hline(
                    y=thr_last,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"임계치: {thr_last:.4f}"
                )
        
        fig.update_layout(
            title="실시간 이상 탐지 점수",
            height=400,
            xaxis_title="인덱스",
            yaxis_title="이상 점수",
            hovermode="x unified",
            template="plotly_white"
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # 5초 경고 표시
        if time.time() < st.session_state.alert_until:
            alert_placeholder.warning(st.session_state.alert_msg)
        else:
            alert_placeholder.empty()
        
        # 스트리밍 루프
        if st.session_state.stream_running:
            if st.session_state.stream_target is None:
                status_placeholder.warning("⚠️ 먼저 '모니터링 시작' 버튼을 눌러주세요.")
            else:
                target = st.session_state.stream_target
                
                for _ in range(st.session_state.N_PER_RERUN):
                    i = st.session_state.stream_i
                    if i >= len(target):
                        st.session_state.stream_running = False
                        status_placeholder.success("✅ 모니터링 완료 (데이터 끝)")
                        break
                    
                    row = target[i]
                    current_index = st.session_state.START_INDEX + i
                    
                    try:
                        start_time = time.time()
                        response = requests.post(
                            f"{st.session_state.base_url}/predict",
                            json={"values": row.tolist()},
                            timeout=10
                        )
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.last_result = result
                            
                            if result.get("status") == "ready":
                                score = float(result.get("score", 0.0))
                                thr = float(result.get("threshold", 0.0))
                                is_anom = bool(result.get("is_anomaly", False))
                                over_thr = (thr != 0.0 and score > thr)
                                
                                # 상태 표시
                                if is_anom or over_thr:
                                    status_placeholder.markdown(
                                        f'<div class="status-warning">⚠️ 이상 감지! Index: {current_index} | 점수: {score:.4f} > 임계치: {thr:.4f}</div>',
                                        unsafe_allow_html=True
                                    )
                                    # 5초 경고
                                    st.session_state.alert_until = time.time() + 5.0
                                    st.session_state.alert_msg = f"🚨 이상 감지 @ Index={current_index} | 점수={score:.4f}"
                                    st.toast(st.session_state.alert_msg, icon="🚨")
                                else:
                                    status_placeholder.markdown(
                                        f'<div class="status-normal">✅ 정상 | Index: {current_index} | 점수: {score:.4f} (지연: {end_time-start_time:.2f}초)</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                st.session_state.history.append({
                                    "index": current_index,
                                    "score": score,
                                    "threshold": thr,
                                    "is_anomaly": is_anom,
                                    "over_threshold": bool(over_thr),
                                })
                            else:
                                status_placeholder.info(f"⏳ 모델 준비 중... ({result.get('progress', '?')})")
                        else:
                            st.error(f"❌ 서버 오류: {response.status_code}")
                            st.session_state.stream_running = False
                            break
                    
                    except Exception as e:
                        st.session_state.stream_running = False
                        status_placeholder.error(f"❌ 오류 발생: {e}")
                        break
                    
                    st.session_state.stream_i = i + 1
                    time.sleep(st.session_state.SLEEP_SEC)
                
                st.rerun()
        
        # 최근 결과 표시
        st.markdown("---")
        st.markdown("### 📋 최근 예측 결과")
        if st.session_state.last_result:
            result_cols = st.columns(4)
            with result_cols[0]:
                st.metric("상태", st.session_state.last_result.get("status", "N/A"))
            with result_cols[1]:
                st.metric("점수", f"{st.session_state.last_result.get('score', 0):.6f}")
            with result_cols[2]:
                st.metric("임계치", f"{st.session_state.last_result.get('threshold', 0):.6f}")
            with result_cols[3]:
                is_anom = st.session_state.last_result.get('is_anomaly', False)
                st.metric("이상 여부", "⚠️ 이상" if is_anom else "✅ 정상")
            
            with st.expander("상세 JSON 보기"):
                st.json(st.session_state.last_result)
        else:
            st.info("아직 예측 결과가 없습니다.")
    
    # ML 모드
    else:
        st.markdown("### 🎯 머신러닝 기반 실시간 예측")
        st.info("ML 모드는 곧 업데이트됩니다. 현재는 DL 모드를 사용해주세요.")

# =====================================================================
# TAB 2: SHAP 분석
# =====================================================================
with tab2:
    st.markdown("## 🔍 SHAP 중요도 분석")
    st.info("📌 SHAP 분석은 모델의 예측에 각 피처가 얼마나 기여했는지를 보여줍니다.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎯 분석 실행")
        
        if st.button("🚀 SHAP 분석 시작", use_container_width=True, type="primary", key="shap_start"):
            with st.spinner("🔄 SHAP 계산 중... (최대 10분 소요)"):
                try:
                    response = requests.post(
                        f"{st.session_state.base_url}/analyze",
                        timeout=600
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "success":
                            st.session_state.shap_results = data
                            st.session_state.shap_completed = True
                            st.success("✅ SHAP 분석 완료!")
                        else:
                            st.error(f"분석 실패: {data}")
                    else:
                        st.error(f"HTTP 오류: {response.status_code}")
                
                except Exception as e:
                    st.error(f"❌ SHAP 분석 실패: {e}")
        
        st.markdown("---")
        
        if st.session_state.shap_completed:
            st.success("✅ 분석 완료")
            if st.button("🔄 재분석", use_container_width=True):
                st.session_state.shap_completed = False
                st.session_state.shap_results = None
                st.rerun()
        else:
            st.warning("⏳ 분석 대기 중")
    
    with col2:
        st.markdown("### 📊 분석 결과")
        
        if st.session_state.shap_results:
            data = st.session_state.shap_results
            imp = data.get("importance", {})
            
            df_imp = pd.DataFrame([
                {"feature": k, "importance": v}
                for k, v in imp.items()
            ])
            df_imp["abs"] = df_imp["importance"].abs()
            df_imp = df_imp.sort_values("abs", ascending=False)
            
            # Top N 슬라이더
            topn = st.slider("상위 N개 피처 표시", 5, 38, 15, 1)
            df_top = df_imp.head(topn).iloc[::-1]
            
            # 막대 차트
            fig = px.bar(
                df_top,
                x="importance",
                y="feature",
                orientation="h",
                title=f"🎯 SHAP 중요도 상위 {topn}개 피처",
                color="importance",
                color_continuous_scale="RdYlGn_r"
            )
            fig.update_layout(
                height=max(400, topn * 25),
                template="plotly_white",
                xaxis_title="중요도",
                yaxis_title="피처명"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 상세 데이터
            with st.expander("📋 전체 중요도 데이터 보기"):
                st.dataframe(
                    df_imp.drop(columns=["abs"]),
                    use_container_width=True,
                    hide_index=True
                )
                st.json(data)
        else:
            st.info("왼쪽의 '🚀 SHAP 분석 시작' 버튼을 눌러 분석을 시작하세요.")

# =====================================================================
# TAB 3: 시스템 정보
# =====================================================================
with tab3:
    st.markdown("## 📊 시스템 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 설정 정보")
        info_data = {
            "서버 URL": st.session_state.base_url,
            "선택 머신": selected_machine,
            "분석 모델": st.session_state.model_type,
            "현재 단계": st.session_state.workflow_stage,
            "SHAP 완료": "✅" if st.session_state.shap_completed else "❌",
            "모니터링 활성": "🟢" if st.session_state.stream_running else "🔴"
        }
        for k, v in info_data.items():
            st.metric(k, v)
    
    with col2:
        st.markdown("### 📈 통계")
        stats_data = {
            "총 예측 횟수": len(st.session_state.history),
            "이상 감지 횟수": len([x for x in st.session_state.history if x.get('is_anomaly', False)]),
            "현재 인덱스": st.session_state.stream_i,
            "경고 활성": "🚨" if time.time() < st.session_state.alert_until else "✅"
        }
        for k, v in stats_data.items():
            st.metric(k, v)
    
    st.markdown("---")
    
    # API 테스트
    st.markdown("### 🧪 API 테스트")
    st.code(f"{st.session_state.base_url}/docs", language="text")
    st.code(f"{st.session_state.base_url}/health", language="text")
    
    st.markdown("#### cURL 예시")
    st.code(f"""curl -X POST "{st.session_state.base_url}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"values":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}}'""", language="bash")
    
    st.code(f"""curl -X POST "{st.session_state.base_url}/analyze" """, language="bash")

# =====================================================================
# 푸터
# =====================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    🚀 <strong>Keroro 서버 모니터링 시스템</strong> | 
    Powered by Streamlit & FastAPI | 
    <a href="{}/docs" target="_blank">API 문서</a>
</div>
""".format(st.session_state.base_url), unsafe_allow_html=True)
