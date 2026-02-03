import torch
import torch.nn as nn
import shap
import numpy as np

# SHAP 연산을 위한 점수 재계산 모델
class ScoreModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # x shape: [Batch, Win, Dim]
        output, series, prior, _ = self.backbone(x)
        
        # Reconstruction Error (MSE) 기반 기여도 분석
        # SHAP은 스칼라 출력을 선호하므로 윈도우 평균 점수를 산출합니다.
        recon_loss = torch.mean((x - output) ** 2, dim=-1) # [Batch, Win]
        score = recon_loss.mean(dim=-1) # [Batch]
        return score.unsqueeze(-1) # [Batch, 1]

def get_shap_explanation(model, window_data, feature_names, device):
    """
    현재 윈도우 데이터에 대해 SHAP 기여도를 계산하여 상위 5개 원인을 반환합니다.
    """
    try:
        # 1. 분석용 모델 준비
        score_model = ScoreModel(model).to(device).eval()
        
        # 2. 배경 데이터(Background) 설정 
        # 실시간성 보장을 위해 현재 윈도우의 평균값을 기준점으로 잡습니다.
        background = window_data.mean(dim=1, keepdim=True).repeat(1, 100, 1).to(device)
        
        # 3. Explainer 설정 (Gradient 기반)
        explainer = shap.GradientExplainer(score_model, background)
        
        # 4. SHAP 값 계산
        shap_values = explainer.shap_values(window_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # [1, 100, 38] -> [38] (각 피처별 윈도우 전체 기여도 평균)
        importance = np.abs(shap_values[0]).mean(axis=0)
        
        # 5. 상위 5개 추출
        top_indices = np.argsort(importance)[::-1][:5]
        
        reasons = [
            {
                "feature": str(feature_names[i]),
                "importance": float(importance[i])
            }
            for i in top_indices
        ]
        
        return reasons
    except Exception as e:
        print(f"⚠️ SHAP 분석 중 오류 발생: {e}")
        return []