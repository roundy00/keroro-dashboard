import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from pydantic import BaseModel

# 모델 및 유틸 임포트
from model.AnomalyTransformer import AnomalyTransformer
from utils.explain import get_shap_explanation

app = FastAPI()

# ✅ 프론트엔드 연동을 위한 CORS 설정 (필수!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 전역 설정 ---
THRESHOLD = 0.50546515 
WIN_SIZE = 100
ENC_IN = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    'cpu_r','load_1','load_5','load_15','mem_shmem','mem_u','mem_u_e','total_mem',
    'disk_q','disk_r','disk_rb','disk_svc','disk_u','disk_w','disk_wa','disk_wb',
    'si','so','eth1_fi','eth1_fo','eth1_pi','eth1_po','tcp_tw','tcp_use',
    'active_opens','curr_estab','in_errs','in_segs','listen_overflows','out_rsts',
    'out_segs','passive_opens','retransegs','tcp_timeouts','udp_in_dg','udp_out_dg',
    'udp_rcv_buf_errs','udp_snd_buf_errs'
]

# --- 모델 로드 ---
model = AnomalyTransformer(win_size=WIN_SIZE, enc_in=ENC_IN, c_out=ENC_IN).to(DEVICE)
checkpoint_path = "checkpoints/machine-1-1/SMD_machine-1-1_checkpoint.pth"

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    # 가중치 키 이름이 module.로 시작할 경우 대응
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("✅ Model loaded successfully.")
else:
    print(f"❌ Checkpoint not found at {checkpoint_path}")

scaler = joblib.load("scaler_machine-1-1.pkl")
data_buffer = deque(maxlen=WIN_SIZE)

class SensorData(BaseModel):
    values: list

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

@app.post("/predict")
async def predict(data: SensorData):
    if len(data.values) != ENC_IN:
        raise HTTPException(status_code=400, detail="Input dimension mismatch.")

    scaled_val = scaler.transform([data.values])[0]
    data_buffer.append(scaled_val)
    
    if len(data_buffer) < WIN_SIZE:
        return {"status": "collecting", "is_anomaly": False}

    input_tensor = torch.FloatTensor(list(data_buffer)).unsqueeze(0).to(DEVICE)
    score = calculate_anomaly_score(input_tensor)
    is_anomaly = score > THRESHOLD
    
    # ✅ 이상 발생 시 원인 분석(SHAP) 실행
    reasons = []
    if is_anomaly:
        reasons = get_shap_explanation(model, input_tensor, FEATURE_NAMES, DEVICE)
    
    return {
        "status": "ready",
        "score": float(score),
        "is_anomaly": bool(is_anomaly),
        "reasons": reasons  # [{feature: 'cpu_r', importance: 0.12}, ...]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
