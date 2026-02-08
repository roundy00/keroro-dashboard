import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import pickle
import shap
import warnings
from fastapi import FastAPI, HTTPException
from collections import deque
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request


# 모델 파일 임포트
from model.AnomalyTransformer import AnomalyTransformer

# SHAP 및 관련 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

app = FastAPI()

# 정적 파일/템플릿 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"ok": True}


# --- [설정 영역] ---
THRESHOLD = 0.50546515
WIN_SIZE = 100
ENC_IN = 38
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

print("🔥 Using device:", DEVICE)


# 파일 경로 (본인 환경에 맞게 수정 필요)
DATA_DIR = "/Users/hri/east/backend/data"
MACHINE = "machine-1-1"
CKPT_PATH = "checkpoints/machine-1-1/SMD_machine-1-1_checkpoint.pth"
SCALER_PATH = "scaler_machine-1-1.pkl"

# ✅ analyze에서 읽을 파일: NPY로 고정 (PKL 지옥 탈출)
# 파일 이름 예: machine-1-1_test.npy
NPY_TEST_PATH = os.path.join(DATA_DIR, f"{MACHINE}_test.npy")
PKL_TEST_PATH = os.path.join(DATA_DIR, f"{MACHINE}_test.pkl")  # 혹시 남아있을 경우 대비(자동 변환용)

# 피처 이름 정의
new_column_names = [
    'cpu_r','load_1','load_5','load_15','mem_shmem','mem_u','mem_u_e','total_mem',
    'disk_q','disk_r','disk_rb','disk_svc','disk_u','disk_w','disk_wa','disk_wb',
    'si','so','eth1_fi','eth1_fo','eth1_pi','eth1_po','tcp_tw','tcp_use',
    'active_opens','curr_estab','in_errs','in_segs','listen_overflows','out_rsts',
    'out_segs','passive_opens','retransegs','tcp_timeouts','udp_in_dg','udp_out_dg',
    'udp_rcv_buf_errs','udp_snd_buf_errs'
]

# --- [모델 로드 함수] ---
def load_trained_model():
    model = AnomalyTransformer(
        win_size=WIN_SIZE,
        enc_in=ENC_IN,
        c_out=ENC_IN,
        e_layers=3
    ).to(DEVICE)

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        state = ckpt.get("state_dict", ckpt)
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        model.eval()
        print(f"✅ Model loaded from {CKPT_PATH}")
    else:
        print(f"❌ Warning: Checkpoint not found at {CKPT_PATH}")
    return model

# 전역 모델 및 스케일러 초기화
model = load_trained_model()
scaler = joblib.load(SCALER_PATH)
data_buffer = deque(maxlen=WIN_SIZE)

# --- [SHAP 분석을 위한 ScoreModel 클래스] ---
def kl_divergence(p, q, eps=1e-8):
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=-1)

class ScoreModel(torch.nn.Module):
    def __init__(self, backbone, alpha=1.0, beta=1.0):
        super().__init__()
        self.backbone = backbone
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        output, series, prior, _ = self.backbone(x)
        recon = torch.mean((output - x) ** 2, dim=(1, 2))

        dis_list = []
        for s, p in zip(series, prior):
            kl = kl_divergence(s, p).mean(dim=(1, 2))
            dis_list.append(kl)
        assoc = torch.stack(dis_list, dim=0).mean(dim=0)

        score = self.alpha * recon + self.beta * assoc
        return score.unsqueeze(-1)

score_model = ScoreModel(model).to(DEVICE).eval()

# --- [실시간 예측용 보조 함수] ---
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def calculate_anomaly_score(window_tensor):
    with torch.no_grad():
        output, series, prior, _ = model(window_tensor)
        loss = torch.mean((window_tensor - output) ** 2, dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for i in range(len(series)):
            series_loss += my_kl_loss(series[i], prior[i])
            prior_loss += my_kl_loss(prior[i], series[i])
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        score = metric * loss
        return score[0, -1].item()

# --- [API 엔드포인트] ---
class SensorData(BaseModel):
    values: list

@app.post("/predict")
async def predict(data: SensorData):
    """실시간 데이터 포인트 수집 및 이상 탐지"""
    if len(data.values) != ENC_IN:
        raise HTTPException(status_code=400, detail=f"Input must have {ENC_IN} features.")

    scaled_val = scaler.transform([data.values])[0]
    data_buffer.append(scaled_val)

    if len(data_buffer) < WIN_SIZE:
        return {"status": "collecting", "progress": f"{len(data_buffer)}/{WIN_SIZE}", "is_anomaly": False}

    input_tensor = torch.from_numpy(np.array(data_buffer)).float().unsqueeze(0).to(DEVICE)
    score = calculate_anomaly_score(input_tensor)
    is_anomaly = score > THRESHOLD

    return {
        "status": "ready",
        "score": float(score),
        "threshold": THRESHOLD,
        "is_anomaly": bool(is_anomaly)
    }

# ============================================================
# ✅ NPY 기반 데이터 로드/윈도잉
# ============================================================
def ensure_test_npy_exists():
    """
    machine-1-1_test.npy가 없으면, 같은 폴더의 pkl이 존재할 때 자동 변환해서 만들어줌.
    """
    if os.path.exists(NPY_TEST_PATH):
        return

    if os.path.exists(PKL_TEST_PATH):
        with open(PKL_TEST_PATH, "rb") as f:
            x_raw = pickle.load(f)
        x_raw = np.asarray(x_raw, dtype=np.float32)
        np.save(NPY_TEST_PATH, x_raw)
        print(f"✅ Converted PKL -> NPY: {PKL_TEST_PATH} -> {NPY_TEST_PATH}")
        return

    raise FileNotFoundError(f"Neither NPY nor PKL test file found. Need: {NPY_TEST_PATH} (or {PKL_TEST_PATH})")

def load_test_array():
    """
    (T, 38) ndarray 로드
    """
    ensure_test_npy_exists()
    x_raw = np.load(NPY_TEST_PATH)
    x_raw = np.asarray(x_raw, dtype=np.float32)

    # shape 검증
    if x_raw.ndim != 2 or x_raw.shape[1] != ENC_IN:
        raise ValueError(f"test array must be shape (T,{ENC_IN}), got {tuple(x_raw.shape)}")

    return x_raw

def load_windows_from_raw(x_raw, window_length=100, step=None, multiply_20=True):
    """
    x_raw: np.ndarray (T,38) 또는 torch.Tensor (T,38)
    return: torch.Tensor (N,100,38)
    """
    if isinstance(x_raw, np.ndarray):
        x = torch.from_numpy(x_raw).float()
    else:
        x = torch.tensor(x_raw, dtype=torch.float32)

    if multiply_20:
        x = x * 20.0

    if x.ndim != 2 or x.shape[1] != ENC_IN:
        raise ValueError(f"raw data must be (T,{ENC_IN}), got {tuple(x.shape)}")

    T = x.shape[0]
    step = step or window_length

    if T < window_length:
        raise ValueError(f"not enough timesteps to make a window: T={T}, win={window_length}")

    windows = []
    for st in range(0, T - window_length + 1, step):
        windows.append(x[st:st + window_length])  # (100,38)

    if len(windows) == 0:
        raise ValueError(f"not enough windows produced: T={T}, win={window_length}, step={step}")

    return torch.stack(windows, dim=0)  # (N,100,38)

def make_feature_summary_X(windows, summary="mean"):
    if summary == "mean":
        return windows.mean(dim=1)          # (N,38)
    if summary == "last":
        return windows[:, -1, :]            # (N,38)
    if summary == "maxabs":
        return windows.abs().max(dim=1).values
    raise ValueError("unknown summary")

def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    sv = np.asarray(shap_values)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    if sv.ndim == 4 and sv.shape[1] == 1:
        sv = sv[:, 0, :, :]
    sv = np.squeeze(sv)
    return sv

@app.post("/analyze")
async def analyze():
    """
    ✅ 기존엔 384 windows 고정으로 잘라서 에러났는데,
    이제는 '현재 가진 window 개수'에 맞게 bg/X 개수를 자동 조절해서
    데이터가 적어도 절대 400을 안 띄우도록 만든다.
    """
    try:
        # 0) test array 로드 (npy)
        x_raw = load_test_array()

        # 1) windows 생성
        windows = load_windows_from_raw(
            x_raw, window_length=WIN_SIZE, step=WIN_SIZE, multiply_20=True
        )  # (N,100,38)

        # 2) mean summary
        X_all = make_feature_summary_X(windows, "mean")  # (N,38)
        N = X_all.shape[0]

        # 3) ✅ 여기서부터 핵심: 데이터 적어도 자동 조절
        # 원래: bg=128, X=256 (총 384)
        # 지금: N이 작으면 비율 유지하면서 줄임
        #
        # 최소 조건: bg >= 1, X >= 1
        if N < 2:
            raise ValueError(f"need at least 2 windows to run SHAP, got {N}")

        # 기본 목표
        target_bg = 128
        target_X = 256

        if N >= (target_bg + target_X):
            bg_n = target_bg
            x_n = target_X
        else:
            # N이 부족할 경우:
            # bg를 N의 1/3, X를 나머지로 두되,
            # bg 최소 1, X 최소 1 보장
            bg_n = max(1, N // 3)
            x_n = max(1, N - bg_n)

        bg = X_all[:bg_n].to(DEVICE)                 # (bg_n,38)
        X  = X_all[bg_n:bg_n + x_n].to(DEVICE)       # (x_n,38)

        # 혹시 slicing 때문에 X가 비었으면 재조정
        if X.shape[0] == 0:
            bg_n = max(1, N - 1)
            x_n = 1
            bg = X_all[:bg_n].to(DEVICE)
            X  = X_all[bg_n:bg_n + x_n].to(DEVICE)

        # 4) (N,38) -> (N,100,38)
        def to_window(x):
            return x.unsqueeze(1).repeat(1, WIN_SIZE, 1)

        bg_w = to_window(bg)  # (bg_n,100,38)
        X_w  = to_window(X)   # (x_n,100,38)

        # 5) SHAP
        explainer = shap.GradientExplainer(score_model, bg_w)
        shap_values = normalize_shap_values(explainer.shap_values(X_w))

        # 6) time축 평균 -> (x_n,38)
        sv_BK = np.abs(shap_values).mean(axis=1)

        # 7) 전체 평균 중요도 (38,)
        sv = sv_BK.mean(axis=0)

        importance = {name: float(val) for name, val in zip(new_column_names, sv)}

        return {
            "status": "success",
            "importance": importance,
            "machine": MACHINE,
            "n_windows": int(N),
            "bg_n": int(bg_n),
            "x_n": int(x_n),
            "test_path": NPY_TEST_PATH
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
