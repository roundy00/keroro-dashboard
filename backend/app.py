#프론트 초안 : 다영님 확장자 파일로 수정된 것으로 앎.
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="Keroro Anomaly Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Keroro Anomaly Dashboard")
st.caption("FastAPI(/predict, /analyze) 서버와 연결해 실시간 점수 + SHAP 중요도를 확인합니다.")

# ------------------------------------------------------------
# 서버 URL 입력 (localhost / ngrok)
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")

    default_base = st.session_state.get("base_url", "http://localhost:8000")
    base_url = st.text_input(
        "FastAPI Base URL",
        value=default_base,
        help="예: http://localhost:8000 또는 https://xxxx.ngrok-free.app",
    ).strip().rstrip("/")

    st.session_state["base_url"] = base_url

    st.divider()
    st.subheader("🔌 연결 상태")
    if st.button("Health Check", use_container_width=True):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                st.success(f"✅ OK: {r.json()}")
            else:
                st.error(f"❌ HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            st.error(f"❌ 연결 실패: {e}")

# ------------------------------------------------------------
# 피처 이름(네 main.py에 맞춤) - TAB2용
# ------------------------------------------------------------
FEATURE_NAMES = [
    'cpu_r','load_1','load_5','load_15','mem_shmem','mem_u','mem_u_e','total_mem',
    'disk_q','disk_r','disk_rb','disk_svc','disk_u','disk_w','disk_wa','disk_wb',
    'si','so','eth1_fi','eth1_fo','eth1_pi','eth1_po','tcp_tw','tcp_use',
    'active_opens','curr_estab','in_errs','in_segs','listen_overflows','out_rsts',
    'out_segs','passive_opens','retransegs','tcp_timeouts','udp_in_dg','udp_out_dg',
    'udp_rcv_buf_errs','udp_snd_buf_errs'
]
ENC_IN = 38

# ------------------------------------------------------------
# 세션 상태 초기화
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []  # [{"index":..., "score":..., "threshold":..., "is_anomaly":..., "over_threshold":...}]
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# streaming state
if "stream_running" not in st.session_state:
    st.session_state["stream_running"] = False
if "stream_i" not in st.session_state:
    st.session_state["stream_i"] = 0
if "stream_target" not in st.session_state:
    st.session_state["stream_target"] = None

# alert state
if "alert_until" not in st.session_state:
    st.session_state["alert_until"] = 0.0
if "alert_msg" not in st.session_state:
    st.session_state["alert_msg"] = ""

# ------------------------------------------------------------
# UI 레이아웃: 탭
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 실시간 예측", "🔍 SHAP 분석", "🧪 API 테스트(개발자용)"])

# ============================================================
# TAB 1) 실시간 예측 (/predict)  (NPY -> row -> POST)
# ============================================================
with tab1:
    st.subheader("📈 실시간 예측 (/predict)")
    st.caption("NPY(machine-*.npy)를 START_INDEX부터 한 행씩 /predict로 전송하고, 결과로 그래프를 실시간 갱신합니다.")

    # ---------------------------
    # (A) 설정: 경로/시작점/슬립
    # ---------------------------
    colA, colB, colC, colD = st.columns([2.2, 1, 1, 1])

    with colA:
        file_path = st.text_input(
            "NPY FILE_PATH",
            value=st.session_state.get("FILE_PATH", "/Users/hri/east/backend/data/machine-1-1.npy"),
        )
        st.session_state["FILE_PATH"] = file_path

    with colB:
        start_index = st.number_input(
            "START_INDEX",
            min_value=0,
            value=int(st.session_state.get("START_INDEX", 15800)),
            step=1
        )
        st.session_state["START_INDEX"] = int(start_index)

    with colC:
        sleep_sec = st.number_input(
            "sleep (sec)",
            min_value=0.01,
            value=float(st.session_state.get("SLEEP_SEC", 0.2)),
            step=0.01
        )
        st.session_state["SLEEP_SEC"] = float(sleep_sec)

    with colD:
        n_per_rerun = st.number_input(
            "N_PER_RERUN",
            min_value=1,
            value=int(st.session_state.get("N_PER_RERUN", 5)),
            step=1
        )
        st.session_state["N_PER_RERUN"] = int(n_per_rerun)

    # ---------------------------
    # (B) 컨트롤 버튼 (Start/Stop/Reset)
    # ---------------------------
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        start_btn = st.button("▶️ Start Streaming", use_container_width=True, type="primary")
    with c2:
        stop_btn = st.button("⏹ Stop", use_container_width=True)
    with c3:
        reset_btn = st.button("🧹 Reset (history & cursor)", use_container_width=True)

    if reset_btn:
        st.session_state["stream_running"] = False
        st.session_state["stream_i"] = 0
        st.session_state["stream_target"] = None
        st.session_state["history"] = []
        st.session_state["last_result"] = None
        st.session_state["alert_until"] = 0.0
        st.session_state["alert_msg"] = ""
        st.success("리셋 완료")

    # Start: 데이터 로드 & 슬라이싱
    if start_btn:
        if not os.path.exists(file_path):
            st.error(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        else:
            all_data = np.load(file_path)
            target_data = all_data[int(start_index):]
            st.session_state["stream_target"] = target_data
            st.session_state["stream_i"] = 0
            st.session_state["stream_running"] = True
            st.success(f"📂 Loaded: {all_data.shape} | Streaming from {start_index} (len={len(target_data)})")

    if stop_btn:
        st.session_state["stream_running"] = False

    # ---------------------------
    # (C) API 호출 함수
    # ---------------------------
    def call_predict(v):
        r = requests.post(f"{base_url}/predict", json={"values": v}, timeout=10)
        r.raise_for_status()
        return r.json()

    # ---------------------------
    # (D) placeholders: 항상 존재
    # ---------------------------
    alert_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()

    # ---------------------------
    # (E) ✅ 차트는 "항상" 먼저 그려서 화면에 떠있게 만들기
    # ---------------------------
    df_h = pd.DataFrame(st.session_state["history"]) if len(st.session_state["history"]) > 0 else pd.DataFrame(
        columns=["index", "score", "threshold", "is_anomaly", "over_threshold"]
    )

    fig = go.Figure()

    # score line (비어있어도 trace를 추가해두면 figure가 항상 보임)
    fig.add_trace(go.Scatter(
        x=df_h["index"] if "index" in df_h else [],
        y=df_h["score"] if "score" in df_h else [],
        mode="lines+markers",
        name="score"
    ))

    # anomaly markers
    if len(df_h) > 0 and "is_anomaly" in df_h.columns:
        an = df_h[df_h["is_anomaly"] == True]
        if len(an) > 0:
            fig.add_trace(go.Scatter(
                x=an["index"], y=an["score"],
                mode="markers",
                name="anomaly",
                marker=dict(symbol="x", size=10)
            ))

    # over-threshold markers
    if len(df_h) > 0 and "over_threshold" in df_h.columns:
        over = df_h[df_h["over_threshold"] == True]
        if len(over) > 0:
            fig.add_trace(go.Scatter(
                x=over["index"], y=over["score"],
                mode="markers",
                name="over_threshold",
                marker=dict(size=9, symbol="circle-open")
            ))

    # threshold line (마지막 값)
    if len(df_h) > 0 and "threshold" in df_h.columns and df_h["threshold"].notna().any():
        thr_last = float(df_h["threshold"].iloc[-1])
        fig.add_hline(y=thr_last, line_dash="dash", annotation_text=f"threshold={thr_last:.4f}")

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="index",
        yaxis_title="score",
        hovermode="x unified",
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # (F) 5초 경고 표시 (항상)
    # ---------------------------
    if time.time() < float(st.session_state["alert_until"]):
        alert_placeholder.warning(st.session_state["alert_msg"])
    else:
        alert_placeholder.empty()

    # ---------------------------
    # (G) 스트리밍 루프 (chunk 처리 후 rerun)
    # ---------------------------
    if st.session_state["stream_running"]:
        if st.session_state["stream_target"] is None:
            status_placeholder.warning("START를 눌러 NPY를 로드해줘.")
        else:
            target = st.session_state["stream_target"]

            for _ in range(int(n_per_rerun)):
                i = st.session_state["stream_i"]
                if i >= len(target):
                    st.session_state["stream_running"] = False
                    status_placeholder.info("✅ 스트리밍 끝 (EOF)")
                    break

                row = target[i]
                current_index = int(start_index) + i

                try:
                    start_time = time.time()
                    result = call_predict(row.tolist())
                    end_time = time.time()

                    st.session_state["last_result"] = result

                    if result.get("status") == "ready":
                        score = float(result.get("score", 0.0))
                        thr = float(result.get("threshold", 0.0))
                        is_anom = bool(result.get("is_anomaly", False))
                        over_thr = (thr != 0.0 and score > thr)

                        status_str = "⚠️ 이상" if (is_anom or over_thr) else "✅ 정상"
                        status_placeholder.write(
                            f"[{current_index}] {status_str} | score={score:.4f} / thr={thr:.4f} "
                            f"(latency={end_time-start_time:.3f}s)"
                        )

                        # anomaly 감지 순간 5초 경고
                        if is_anom or over_thr:
                            st.session_state["alert_until"] = time.time() + 5.0
                            st.session_state["alert_msg"] = (
                                f"🚨 ANOMALY DETECTED @ index={current_index} | score={score:.4f} > thr={thr:.4f}"
                            )
                            alert_placeholder.warning(st.session_state["alert_msg"])
                            st.toast(st.session_state["alert_msg"], icon="🚨")

                        st.session_state["history"].append({
                            "index": current_index,
                            "score": score,
                            "threshold": thr,
                            "is_anomaly": is_anom,
                            "over_threshold": bool(over_thr),
                        })
                    else:
                        # 빌드업 중이라도 상태는 보여주기
                        status_placeholder.write(
                            f"⏳ 빌드업 중... ({result.get('progress', '?')}) (Index: {current_index})"
                        )

                except Exception as e:
                    st.session_state["stream_running"] = False
                    status_placeholder.error(f"❌ 오류 발생: {e}")
                    break

                st.session_state["stream_i"] = i + 1
                time.sleep(float(sleep_sec))

            # chunk 처리 후 계속
            st.rerun()

    # ---------------------------
    # (H) 최근 결과 JSON
    # ---------------------------
    st.divider()
    st.markdown("### 최근 결과")
    if st.session_state["last_result"] is None:
        st.info("아직 /predict 결과가 없음")
    else:
        st.json(st.session_state["last_result"])

# ============================================================
# TAB 2) SHAP 분석 (/analyze)
# ============================================================
with tab2:
    st.subheader("🔍 SHAP 분석 (/analyze)")
    st.caption("서버가 내부적으로 pkl을 읽어 SHAP 중요도 38개를 계산한 뒤 JSON으로 내려줍니다.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 실행")
        st.info("버튼을 누르면 서버가 시간이 좀 걸리는 SHAP 계산을 수행합니다.")
        run_analyze = st.button("🚀 SHAP 분석 실행", use_container_width=True, type="primary")

        if run_analyze:
            with st.spinner("SHAP 계산 중... (서버가 무거우면 수십초 걸릴 수 있음)"):
                try:
                    r = requests.post(f"{base_url}/analyze", timeout=600)
                    if r.status_code != 200:
                        st.error(f"HTTP {r.status_code}: {r.text[:500]}")
                    else:
                        data = r.json()
                        if data.get("status") != "success":
                            st.error(f"응답 status가 success가 아님: {data}")
                        else:
                            st.session_state["shap_result"] = data
                            st.success("✅ SHAP 분석 완료!")
                except Exception as e:
                    st.error(f"/analyze 호출 실패: {e}")

    with col2:
        st.markdown("### 결과")
        data = st.session_state.get("shap_result")
        if not data:
            st.info("아직 결과 없음")
        else:
            imp = data.get("importance", {})
            df_imp = pd.DataFrame([{"feature": k, "importance": v} for k, v in imp.items()])
            df_imp["abs"] = df_imp["importance"].abs()
            df_imp = df_imp.sort_values("abs", ascending=False)

            topn = st.slider("Top N", 5, 38, 10, 1)
            df_top = df_imp.head(topn).iloc[::-1]

            fig = px.bar(
                df_top,
                x="importance",
                y="feature",
                orientation="h",
                title=f"SHAP Feature Importance (Top {topn})"
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("전체 중요도 JSON / Table"):
                st.json(data)
                st.dataframe(df_imp.drop(columns=["abs"]), use_container_width=True, hide_index=True)

# ============================================================
# TAB 3) 개발자용 API 테스트
# ============================================================
with tab3:
    st.subheader("🧪 API 테스트")
    st.write("브라우저에서 바로 확인:")
    st.code(f"{base_url}/docs", language="text")
    st.code(f"{base_url}/health", language="text")

    st.markdown("### curl 예시")
    st.code(
        f"""curl -X POST "{base_url}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"values":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}}'""",
        language="bash"
    )
    st.code(
        f"""curl -X POST "{base_url}/analyze" """,
        language="bash"
    )

st.divider()
st.caption("Tip: ngrok을 쓰면 Streamlit도 8501 포트로 열어서 팀원에게 링크 공유 가능!")
