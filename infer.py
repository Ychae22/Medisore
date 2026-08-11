# infer.py — 욕창 단계 분류기 A: 추론 전용 (TA 인계용, 자체 완결)
# v2 · 2026-08-06 · 데이터 B(YOLO 위치검출) 드랍 반영
# ===========================================================================
# TA(웹/앱 개발) 사용법:
#   from infer import predict
#   result = predict("wound.jpg")          # 파일경로 또는 PIL.Image
#
#   # 정상 판정
#   # {"stage": 3, "label": "Stage 3", "confidence": 0.86,
#   #  "message": "욕창 3단계로 보입니다.",
#   #  "probs": {"Stage 1":0.02,"Stage 2":0.05,"Stage 3":0.86,"Stage 4":0.07},
#   #      ↑ probs 키에는 공백이 있습니다. probs["Stage1"] 은 KeyError.
#   #  "low_confidence": False, "reject": False}
#
#   # 확신도 낮음 (재촬영 유도)
#   # {"stage": None, "label": None, "confidence": 0.41,
#   #  "message": "확신도가 낮습니다 — 상처가 화면에 크게 담기도록 다시 촬영해 주세요.",
#   #  "probs": {...}, "low_confidence": True, "reject": True}
#
# 필요 패키지: torch, torchvision, pillow   (pip install torch torchvision pillow)
# 필요 파일  : best_efficientnet_b0.pt  (이 파일과 같은 폴더 또는 MODEL_PATH 지정)
#
# 입력 전제: 상처가 화면을 크게 채운 사진 1장.
#            앞단에 위치검출 모델은 없습니다. 촬영 화면에서 사용자가 상처를
#            크게 담아 찍도록 안내해 주세요.
#            전처리(224 리사이즈 + 정규화)는 학습과 동일하게 이 파일 안에서 처리합니다.
#
# 한계 : 이 모델은 "욕창인지 아닌지"를 판정하지 못합니다. 어떤 사진이든
#        1~4단계 중 하나로 답합니다. low_confidence 는 '욕창 아님' 신호가
#        아니라 '모델이 헷갈림' 신호입니다.
# ===========================================================================
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ── 설정 ────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "best_efficientnet_b0.pt"
CONFIDENCE_THRESHOLD = 0.55     # 이 값 미만이면 재촬영 안내 (val 로 튜닝 가능)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


# 모델은 최초 1회만 로드해서 재사용(웹서버에서 매 요청마다 로드 금지)
_model = None


def _get_model():
    global _model
    if _model is None:
        ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        m = _build(ck.get("model_name", "efficientnet_b0"))
        m.load_state_dict(ck["model"])
        m.eval().to(DEVICE)
        _model = m
    return _model


@torch.no_grad()
def predict(image):
    """image: 파일경로(str) 또는 PIL.Image → dict 반환"""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")

    x = _tf(image).unsqueeze(0).to(DEVICE)
    probs = F.softmax(_get_model()(x), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    conf = float(conf)
    stage = int(idx) + 1
    probs_d = {CLASS_NAMES[i + 1]: round(float(probs[i]), 3) for i in range(4)}

    if conf < CONFIDENCE_THRESHOLD:
        return {"stage": None, "label": None, "confidence": round(conf, 3),
                "message": LOW_CONF_MSG, "probs": probs_d,
                "low_confidence": True,
                "reject": True}          # reject 는 v1 호환용 별칭

    return {"stage": stage, "label": CLASS_NAMES[stage],
            "confidence": round(conf, 3),
            "message": f"욕창 {stage}단계로 보입니다.", "probs": probs_d,
            "low_confidence": False,
            "reject": False}             # reject 는 v1 호환용 별칭


# CLI 테스트:  python infer.py wound.jpg
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("사용법: python infer.py <이미지경로>")
        sys.exit(0)
    print(json.dumps(predict(sys.argv[1]), ensure_ascii=False, indent=2))
