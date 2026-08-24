# infer.py — 욕창 단계 분류기 (Render 512MB 최적화 버전)
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ── Render 무료 티어 CPU 스레드 최적화 (필수) ──
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ── 설정 ────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "best_efficientnet_b0.pt"
CONFIDENCE_THRESHOLD = 0.55
DEVICE = "cpu"  # Render 무료 서버는 무조건 CPU
# ────────────────────────────────────────────────────────────────────

CLASS_NAMES = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "Stage 4"}
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]
LOW_CONF_MSG = "확신도가 낮습니다 — 상처가 화면에 크게 담기도록 다시 촬영해 주세요."

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IM_MEAN, IM_STD),
])

def _build(name="efficientnet_b0"):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    return m

# 서버 시작 시 미리 로드 (지연 로딩 제거)
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
        ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        m = _build(ck.get("model_name", "efficientnet_b0"))
        m.load_state_dict(ck["model"])
        m.eval().to(DEVICE)
        _model = m
    return _model

# 최초 1회 즉시 로드
try:
    get_model()
except Exception as e:
    print(f"[경고] 모델 사전 로드 대기: {e}")

@torch.inference_mode()  # no_grad보다 훨씬 빠르고 메모리를 적게 먹음
def predict(image):
    """image: 파일경로(str) 또는 PIL.Image → dict 반환"""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")

    # 스마트폰 대용량 이미지 사전 리사이징 (메모리 절약)
    image.thumbnail((512, 512))

    x = _tf(image).unsqueeze(0).to(DEVICE)
    probs = F.softmax(get_model()(x), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    conf = float(conf)
    stage = int(idx) + 1
    probs_d = {CLASS_NAMES[i + 1]: round(float(probs[i]), 3) for i in range(4)}

    if conf < CONFIDENCE_THRESHOLD:
        return {"stage": None, "label": None, "confidence": round(conf, 3),
                "message": LOW_CONF_MSG, "probs": probs_d,
                "low_confidence": True,
                "reject": True}

    return {"stage": stage, "label": CLASS_NAMES[stage],
            "confidence": round(conf, 3),
            "message": f"욕창 {stage}단계로 보입니다.", "probs": probs_d,
            "low_confidence": False,
            "reject": False}
