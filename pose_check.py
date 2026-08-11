# -*- coding: utf-8 -*-
"""
욕창 프로젝트 · 누운 자세 판정 (MediaPipe)

기준 문서 : position_criteria_crosshair_KR.xlsx
촬영 조건 : 천장 카메라 · 머리가 위쪽 · 사진 1장
계산 방식 : 2D(x·y)만 사용. MediaPipe z(깊이)는 추정값이라 판정에 쓰지 않습니다.

사진 1장 → 관절 점 추출 → 자세 자동 판별(앙와위/측위) → 항목별 판정
        → 정상(초록)/주의(노랑)/불량(빨강) → 문제 부위만 색칠한 결과 이미지 저장

▶ 판정 기준을 바꾸려면 아래 '★ 판정 기준표' 블록만 고치면 됩니다.
"""
import argparse
import json
import math
import os
import sys

# MediaPipe/absl/oneDNN 로그 억제 — mediapipe import 전에 설정해야 먹힘
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import mediapipe as mp


# ---------------------------------------------------------------- 한글 경로 입출력
def imread_u(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------- 관절 인덱스
POINTS = {
    "nose": 0,        "l_eye": 2,       "r_eye": 5,
    "l_ear": 7,       "r_ear": 8,       # 측위 ① 몸통 일직선에 사용
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13,    "r_elbow": 14,
    "l_wrist": 15,    "r_wrist": 16,
    "l_hip": 23,      "r_hip": 24,
    "l_knee": 25,     "r_knee": 26,
    "l_ankle": 27,    "r_ankle": 28,
    "l_heel": 29,     "r_heel": 30,
    "l_foot": 31,     "r_foot": 32,   # 발가락 끝(foot index)
}
CORE = ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_knee", "r_knee"]

GREEN = (80, 200, 80)    # BGR
YELLOW = (20, 170, 255)
RED = (60, 60, 230)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
COLOR = {"green": GREEN, "yellow": YELLOW, "red": RED, "info": GRAY}
LABEL = {"green": "정상", "yellow": "주의", "red": "불량", "info": "참고"}
RANK = {"green": 0, "yellow": 1, "red": 2}


# ==============================================================================
#  ★ 판정 기준표 — 기준을 바꿀 일이 있으면 이 블록만 고치면 됩니다 ★
# ------------------------------------------------------------------------------
#  기준 문서: position_criteria_crosshair_KR.xlsx
#
#  네 가지 형식이 있습니다.
#    1) 구간형   {"green": [(하한, 상한), ...], "yellow": [...]}
#                → 초록에도 노랑에도 안 들면 자동으로 빨강. 경계값은 초록 우선.
#    2) 최소값형 {"green_min": a, "yellow_min": b}
#                → 값 >= a 초록 / >= b 노랑 / 그 미만 빨강. (비율 항목)
#    3) 상한형   {"red_min": a}
#                → 값 >= a 빨강 / 그 미만 초록. 노랑 없는 이진 항목.
#    4) 스위치형 {"키": True/False}
#                → 좌우 방향 같은 참·거짓 설정.
#
#  단위는 각도(°) 또는 비율(단위 없음). 각 항목 주석에 적어 뒀습니다.
# ==============================================================================

# ── 앙와위 (바로 누움) ─────────────────────────────────────────────────────────
기준_앙와위 = {
    # 항목                     정상(초록)             주의(노랑)
    # ° 세로축 대비 대각선 누움.
    #   🚩 임의 — SLP 압력 1,485프레임에서 천골압과 무상관(rho +0.006, p=0.83).
    #   다만 실측 117장이 전부 정상 구간이라 완화해도 영향이 0이라 그대로 뒀습니다.
    #   실제 환자는 더 비스듬히 누울 수 있어 안전판으로 남깁니다.
    "① 몸통 정렬":   {"green": [(0, 10)],  "yellow": [(10, 20)]},
    # ° 코가 중심선에서 벗어난 각.
    #   ✅ 실측 1,485프레임 — 뒤통수 압력이 1.24 → 1.46 → 1.63 → 1.87배로 단조 증가
    #      (Spearman rho +0.131, p=5e-7). 뚜렷한 계단은 없어 10/20 경계가 무난합니다.
    "② 머리 쏠림":   {"green": [(0, 10)],  "yellow": [(10, 20)]},
    # ° 어깨선 기울기.
    #   ❌ 실측 1,485프레임 — 어깨 압력과 무상관(rho +0.006, p=0.83).
    #      0~4°와 15~20°의 어깨 압력이 같아 8° 경계를 정당화할 근거가 없었습니다.
    #      → 8/15 에서 15/25 로 완화(2026-08-05). 항목 정상률 86.7% → 97.8%.
    #   ※ SLP 앙와위는 반듯한 자세에 치우쳐(0~4°가 1485장 중 980장) 극단값 표본이 적습니다.
    #      "무관"이 아니라 "이 범위에선 차이가 안 보인다"가 정확한 표현입니다.
    "③ 어깨 수평":   {"green": [(0, 15)],  "yellow": [(15, 25)]},
    # ° 골반선 기울기.
    #   △ 실측 1,485프레임 — 좌우 압력 비대칭과 상관은 있으나 약함(rho +0.087, p=8e-4).
    #      8°가 특별한 지점은 아니지만 방향은 맞아 값을 유지합니다.
    "④ 골반 수평":   {"green": [(0, 8)],   "yellow": [(8, 15)]},
    # ° 눈선 기울기. 🚩 임의 · 미검증 — SLP 정답 관절에 눈 좌표가 없어 압력 검증을 못 했습니다.
    "⑤ 머리 갸우뚱": {"green": [(0, 10)],  "yellow": [(10, 20)]},
    # ° 침대머리(HOB). 근거 EPUAP/NPIAP R11.
    #   미검증 — SLP는 침대가 항상 평평해 이 항목만 검증할 수 없었습니다.
    "⑥ 상체 기울기": {"green": [(0, 30)],  "yellow": [(30, 45)]},
    # ° 이상이면 완전신전. 근거 OpenStax 9.4.
    #   ※ 목적이 바뀌었습니다(2026-08-05). 실측에서 무릎을 굽힐수록 뒤꿈치 압력이 오히려
    #     높았습니다(120~150° 8.74배 vs 178~181° 2.38배, rho −0.186, p=7e-13).
    #     SLP에서 무릎을 굽힌 자세는 발바닥을 매트에 딛는 자세라 그렇습니다.
    #     임상에서 무릎 밑 베개의 목적은 굽히는 것 자체가 아니라 '뒤꿈치를 띄우는 것'이므로,
    #     각도 기준은 그대로 두되 안내 문구를 "무릎 밑에 베개를 받쳐 뒤꿈치가 뜨게" 로 씁니다.
    "⑦ 무릎 신전":   {"red_min": 175},
    # 비율 = 발목간격 / 골반간격.
    #   🚩 임의 · **압력으로는 검증 불가** — 압력매트는 몸과 매트 사이만 잽니다.
    #      이 항목이 막으려는 위험(복사뼈·무릎 안쪽끼리 눌림)이 정확히 그 못 재는 부분입니다.
    #      실측에서 다리를 붙일수록 발목 압력이 '낮게' 나온 것도 같은 이유입니다.
    "⑩ 다리 모임":   {"green_min": 0.7, "yellow_min": 0.5},
}

# ── 측위 (좌·우 공통) ─────────────────────────────────────────────────────────
#   좌측위/우측위는 문서상 수치가 완전히 같아 기준표를 하나로 씁니다.
#   좌·우 구분은 '어느 다리가 아래쪽인지' 판별하는 데만 내부적으로 씁니다.
기준_측위 = {
    "① 몸통 일직선":     {"green": [(0, 25)],  "yellow": [(25, 40)]},   # ° 머리-어깨-골반 꺾임각
    #    ↑ 문서 원안은 정상 <15 / 주의 15~25 였습니다.
    #      머리 위치를 코 → 귀중점으로 바꿔 재고도 측위 10장 실측이 12~45°(중앙 25°)라
    #      원안으로는 9/10이 주의·불량이 됐습니다. 옆으로 누우면 대개 머리를 조금 숙이고
    #      팔을 베기 때문입니다. 실측 중앙값을 경계로 잡아 완화한 값입니다.
    #      (앙와위는 실측 중앙 3°라 이 변경의 영향을 받지 않습니다.)
    #    ⚠ 재검토 대상 — 실측 117장에서 **측위 72장의 59.7%가 주의·불량**입니다.
    #      측위 등급을 사실상 이 항목 하나가 결정하고 있습니다. 다만 정답 관절에 귀 좌표가
    #      없어 압력 검증을 못 했습니다. 측위 사진이 모이면 다시 봐야 합니다.
    "② 측위 각도":       {"green": [(20, 45)], "yellow": [(10, 20), (45, 60)]},
    #    ↑ 문서는 정상 20~45 / 주의 10~20 / 위험 >60·<10 으로만 적혀 있어
    #      45~60 구간이 비어 있었습니다. 정상과 위험 사이이므로 주의로 채웠습니다.
    #    ✅ 실측 2,970프레임으로 검증됨 — 바닥 쪽 대전자 압력이
    #         30~45° 2.67배 → 45~60° 3.27 → 60~75° 4.44 → 75~90° 6.06
    #       으로 오르고, **45°와 60° 경계에 정확히 계단이 있습니다**
    #       (Spearman rho +0.386, p=2e-106). EPUAP/NPIAP의 30° 측위 권고와도 일치.
    #       기준을 옮길 근거가 없습니다. 다만 20° 미만 구간은 n=21로 여전히 표본이 얇습니다.
    "③ 무릎 신전":       {"red_min": 175},                             # ° 아래쪽 다리 기준
    "⑥ 다리 모임(발목)": {"green_min": 0.22, "yellow_min": 0.15},      # 비율 = 발목간격 / 몸통길이
    "⑥ 다리 모임(무릎)": {"green_min": 0.30, "yellow_min": 0.22},      # 비율 = 무릎간격 / 몸통길이
    #    ↑ 지금은 '참고 표시'라 이 두 값은 쓰이지 않습니다(아래 표시설정 참고).
    #      측위는 골반이 겹쳐 보여 골반너비를 자로 못 써서 몸통길이 기준으로 잡은 값이며,
    #      앙와위 9장 실측(골반너비/몸통길이 = 0.38)으로 환산한 추정값입니다.
}

# ── 앙와위 전용 ─────────────────────────────────────────────────────────────
기준_공통 = {
    # 다리 교차(X자 꼬임) — 좌우 x좌표가 뒤집혔는지로 판별. 앙와위에서만 씁니다.
    # 앙와위 9장 전부 '왼발목이 화면에서 더 오른쪽'이 정상으로 확인됨(9/9).
    # 카메라를 반대로 달거나 프론트가 좌우반전하면 False 로 뒤집으세요.
    #   ※ 측위에서는 쓰지 않습니다. 몸이 90°에 가깝게 돌면 좌우 x순서가 원래 뒤집혀서
    #      정상 겹침과 실제 꼬임을 구분할 수 없기 때문입니다(측위 7장 중 4장 오탐 확인).
    "⑨ 다리 교차": {"정상은_왼발목이_더_오른쪽": True},

    # ⑧ 족하수 — 발목 각도. 90°=중립(발이 섬), 작을수록 발이 처진 것.
    #   천장에서는 각도를 직접 못 재고, 발꿈치→발끝이 짧아 보이는 정도로 역산합니다.
    #     투영길이 ÷ 몸통길이 = 발길이_몸통비 × cos(각도)
    #   발끝이 머리 쪽을 향하면(배측굴곡) 문서상 정상이므로 90°로 봅니다.
    #   ※ 2026-08-05 — '자세 그룹으로 올릴까'를 검토했다가 **'지지대' 유지로 결론**냈습니다.
    #     올릴 근거: 실측 앙와위 최대 압력 부위가 천골(2.3배)이 아니라 **뒤꿈치(3.4배)** 였음.
    #     유지한 이유:
    #       ① 실측 117장에서 80° 이상이 6.8%뿐이라 등급에 넣으면 '정상'이 4.3%로 사라집니다.
    #          경계를 60°로 낮춰 봐도 43.6%가 주의가 되어, 등급이 "발판이 없다"만 말하고
    #          정작 고칠 수 있는 자세 신호가 묻힙니다.
    #       ② 그 60°는 실측 중앙값에서 뽑은 값이라 임상 근거가 없습니다.
    #          근거 없는 기준으로 등급을 흔드는 건 ③ 어깨 수평을 완화한 논리와 모순됩니다.
    #       ③ 이제 예상 압력 히트맵이 뒤꿈치를 빨갛게 보여줍니다. 등급에 넣지 않아도 전달됩니다.
    #     대신 **안내 문구를 항목별로 구체화**했습니다(아래 표시설정 참고).
    #     등급은 '지금 당장 고칠 수 있는 것', 히트맵·문구는 '왜 위험한지'로 역할을 나눕니다.
    "⑧ 족하수": {
        "green": [(80, 90)], "yellow": [(70, 80)],      # ° 문서 기준 그대로
        # 발길이 ÷ 몸통길이 — 발이 완전히 눕혀졌을 때의 비율(자 역할).
        #   해부학 추정: 발 길이 = 키의 15.2%, 어깨중점~골반중점 = 키의 28.8%
        #   → 0.152 / 0.288 = 0.528
        #   SLP 102명(발 202개) 실측 투영은 0.152~0.385로 이 값보다 작아,
        #   전원이 어느 정도 처진 상태임을 뜻합니다(발판 없이 누우면 정상적인 현상).
        #   발끝을 쭉 뻗어 침대에 닿게 한 사진이 생기면 그 값으로 바꾸세요.
        "발길이_몸통비": 0.528,
    },
}

# ── 표시 설정 ───────────────────────────────────────────────────────────────
표시설정 = {
    # '지지대' 그룹(⑦ 무릎 신전 · ⑧ 족하수)이 하나라도 정상이 아니면 이 문구를 띄웁니다.
    #   두 항목 모두 "자세가 틀렸다"가 아니라 "베개·발판 같은 도구가 필요하다"는 신호라
    #   전체 등급과 분리해 같은 문구 하나로 묶어 보여줍니다.
    "지지대_안내문구": "보조기구를 사용해주세요",

    # 항목별 안내 — supports["details"] 로 나갑니다. 화면에 한 줄씩 띄우세요.
    #   등급을 내리는 대신 "무엇을 · 왜"를 문장으로 전달합니다.
    #   why 는 실측 압력 근거입니다(앙와위 평균 접촉압 = 1 기준 배수).
    "지지대_문구": {
        "⑦ 무릎 신전": {
            "do": "무릎 밑에 베개를 받쳐 뒤꿈치가 침대에서 살짝 뜨게 해 주세요",
            "why": "다리를 쭉 편 채로 두면 뒤꿈치에 체중이 그대로 실립니다",
        },
        "⑧ 족하수": {
            "do": "발바닥에 받침(발판·단단한 쿠션)을 대어 발이 서도록 해 주세요",
            "why": "받침이 없으면 발이 처지면서 뒤꿈치가 계속 눌립니다 — "
                   "바로 누웠을 때 몸에서 가장 세게 눌리는 곳입니다(실측 꼬리뼈의 1.5배)",
        },
    },

    # ── 간병인용 안내 문구 ──────────────────────────────────────────────────
    #   화면에는 각도를 띄우지 않습니다. 이 문장들이 사용자가 보는 전부입니다.
    #   "순위"는 낮을수록 급한 것 — 실측 압력(앙와위 평균 접촉압=1)을 근거로 잡았습니다.
    #     엉덩이 옆 뼈 5.0(측위) > 뒤꿈치 3.4 > 뒤통수 2.5 > 꼬리뼈 2.3 > 어깨 0.8
    #   ※ 압력 '크기' 기준입니다. 욕창 '발생 빈도'는 천골이 가장 높다는 게 통설이라
    #     축이 다릅니다. 간호 쪽 검토를 한 번 받는 게 좋습니다.
    "안내_최대개수": 3,          # 한 번에 이 개수까지만. 나머지는 접어서 보여주세요

    # 자세 이름도 간병인 말로 — '앙와위·측위'는 임상 용어라 가족은 못 알아듣습니다.
    "자세_사용자표기": {"앙와위": "바로 누움", "좌측위": "왼쪽으로 누움",
                    "우측위": "오른쪽으로 누움", "측위": "옆으로 누움"},
    "자세_문구": {
        "② 측위 각도":   {"bad": "몸이 옆으로 너무 많이 넘어가 있어요",
                        "do": "등 뒤 베개를 낮춰 조금 덜 돌아가게 해 주세요",
                        "why": "엉덩이 옆 뼈가 직접 눌립니다", "순위": 1},
        "② 측위 각도(얕음)": {"bad": "옆으로 조금밖에 안 돌아갔어요",
                        "do": "등 뒤에 베개를 받쳐 조금 더 돌려 주세요",
                        "why": "어중간하면 꼬리뼈가 계속 눌립니다", "순위": 1},
        "② 머리 쏠림":   {"bad": "머리가 한쪽으로 쏠려 있어요",
                        "do": "베개 가운데로 머리를 옮겨 주세요",
                        "why": "뒤통수 한쪽만 계속 눌립니다", "순위": 3},
        "⑤ 머리 갸우뚱": {"bad": "고개가 옆으로 기울어 있어요",
                        "do": "얼굴이 천장을 보도록 고개를 바로 해 주세요",
                        "why": "뒤통수 한쪽에 무게가 쏠립니다", "순위": 3},
        "④ 골반 수평":   {"bad": "엉덩이가 한쪽으로 기울어 있어요",
                        "do": "양쪽 엉덩이가 같은 높이가 되도록 눕혀 주세요",
                        "why": "한쪽 엉덩뼈에만 체중이 실립니다", "순위": 4},
        "⑥ 상체 기울기": {"bad": "침대 머리가 너무 서 있어요",
                        "do": "침대 머리를 조금 더 눕혀 주세요",
                        "why": "몸이 아래로 밀리면서 꼬리뼈가 쓸립니다", "순위": 4},
        "① 몸통 일직선": {"bad": "머리와 몸통이 어긋나 있어요",
                        "do": "베개 높이를 맞춰 머리·어깨·엉덩이가 일직선이 되게 해 주세요",
                        "why": "", "순위": 5},
        "① 몸통 정렬":   {"bad": "몸이 한쪽으로 휘어 있어요",
                        "do": "머리·가슴·엉덩이가 일직선이 되게 곧게 맞춰 주세요",
                        "why": "", "순위": 5},
        "⑨ 다리 교차":   {"bad": "두 다리가 서로 꼬여 있어요",
                        "do": "두 다리를 나란히 풀어 주세요",
                        "why": "복사뼈끼리 눌립니다", "순위": 6},
        "⑩ 다리 모임":   {"bad": "두 다리가 딱 붙어 있어요",
                        "do": "무릎 사이에 얇은 베개를 넣어 한 뼘쯤 띄워 주세요",
                        "why": "무릎 안쪽끼리 눌립니다", "순위": 6},
        "③ 어깨 수평":   {"bad": "어깨가 한쪽으로 기울어 있어요",
                        "do": "양 어깨 높이를 나란히 맞춰 주세요",
                        "why": "", "순위": 7},
    },
    # 정상 항목을 칭찬으로 돌려주는 문장 — 고칠 게 없을 때 화면이 비지 않게 합니다.
    "자세_잘됨": {
        "② 측위 각도": "돌아누운 각도가 적당해요",
        "① 몸통 일직선": "머리·어깨·엉덩이가 일직선이에요",
        "② 머리 쏠림": "머리가 베개 가운데 잘 있어요",
        "⑤ 머리 갸우뚱": "고개가 바르게 놓였어요",
        "④ 골반 수평": "엉덩이 높이가 잘 맞아요",
        "① 몸통 정렬": "몸통이 곧게 잘 놓였어요",
        "⑨ 다리 교차": "다리가 꼬이지 않았어요",
        "⑩ 다리 모임": "다리 간격이 적당해요",
        "③ 어깨 수평": "양 어깨 높이가 잘 맞아요",
    },

    # 측위 '⑥ 다리 모임'을 정상/불량으로 판정할지. False면 숫자만 참고로 보여줍니다.
    #   90° 가까이 돌아누우면 다리가 앞뒤로 포개져 위에서 본 간격이 늘 좁게 나와,
    #   실제 눌림과 구분되지 않습니다. 측위 7장 실측 = 0.057~1.077 (극단으로 갈림).
    #   30° 측위 사진이 모여 기준값을 다듬은 뒤 True 로 바꾸세요.
    "측위_다리모임_판정함": False,
}

# ── 자세 자동 판별 ───────────────────────────────────────────────────────────
#   ※ 어깨가 아니라 '골반'이 주 신호입니다. SLP 117장(앙와위 45 · 측위 72) 실측 —
#       골반너비/몸통길이 : 앙와위 0.322~0.427 / 측위 0.004~0.318 → 두 무리가 안 겹침
#       어깨너비/몸통길이 : 앙와위 0.497~0.738 / 측위 0.012~0.619 → 0.12 겹침
#     어깨는 팔을 올리거나 한쪽이 가리면 그냥 좁아져서, 기준을 어디에 두든 4장이 틀립니다.
#     둘을 섞어봐도(평균·min·max·가중합) 전부 겹쳐서 골반 단독이 최선이었습니다.
자세판별 = {
    # 기준값이 있을 때: 회전각이 이 값 이상이면 측위. '골반' 기준입니다.
    #   실측 — 앙와위 최대 25.6° / 측위 최소 38.3° (여유 12.7°)
    #   어깨 기준이던 때는 앙와위 32.6° / 측위 33.7° 로 여유가 1.1°뿐이었습니다.
    "측위_최소각도": 30,

    # 앙와위 ⑥ 상체 기울기에 "부정확" 경고를 붙일 몸 회전각
    "상체기울기_경고_회전각": 10,

    # 기준값이 없을 때(근사) — 몸통길이 대비 비율이 이 값 미만이면 측위
    #   골반 0.32 : SLP 117장 오분류 0. 다만 경계 여유가 0.005로 얇으니 사진이 모이면 재확인.
    #   어깨 0.52 : 골반이 가려졌을 때만 쓰는 폴백. 최선을 잡아도 117장 중 4장 틀립니다.
    "근사_골반몸통비": 0.32,
    "근사_어깨몸통비": 0.52,

    # 상체와 하체의 회전각 차 — 이 값 이상이면 몸통이 비틀린 것으로 보고 판정 보류.
    #   상체는 앙와위인데 하체만 돌아간 자세는 앙와위 기준표도 측위 기준표도 맞지 않습니다.
    #   ※ 잠정치 — SLP엔 일부러 비튼 자세가 없어 정상군 상한(15.2°)에서 잡았습니다.
    #     반측위 사진이 모이면 재조정하세요.
    "비틀림_보류각": 20,
    # 기준값 없이 근사로 잴 때는 여유를 더 둡니다. 근사는 '평균 체형과의 차이'를 각도로
    # 바꿔버려서, 골반이 좁은 체형이 '하체가 돌아갔다'로 읽힙니다.
    #   실측 25° 기준 — 앙와위 45장 중 2장 보류(1장은 진짜 비틀림, 1장은 체형) · 측위 72장 0장
    #   30°로 올리면 헛보류는 사라지지만 진짜 비틀림(근사 26.3°)도 놓칩니다.
    "비틀림_보류각_근사": 25,

    # 기준값(W0/H0)이 없어도 비틀림은 봐야 하므로, 앙와위 인구 중앙값으로 각도를 근사합니다.
    #   SLP 앙와위 45장 중앙값. 실측 55장 대조 — 절대오차 중앙 0.4°, |오차|≤5° 51/55,
    #   20° 보류 판단은 55/55 일치했습니다. 체형이 평균에서 멀면 오차가 커집니다(최대 13°).
    "앙와위_어깨몸통비": 0.654,
    "앙와위_골반몸통비": 0.376,

    # 측위에서 어느 쪽 몸이 바닥에 닿았는지(=아래쪽 다리) 판별하는 설정.
    #   판별 신호 = (코.x - 귀중점.x) / 몸통길이
    #   실측 10장 — -0.16~-0.08 과 +0.13 으로 뚜렷하게 두 무리로 갈립니다.
    #   ※ MediaPipe visibility 는 못 씁니다. 좌우 차이가 0.000~0.005 로 잡음 수준입니다.
    "좌우판별_최소차이": 0.05,          # 이보다 작으면 얼굴 신호로는 판별 보류
    "코가_화면오른쪽이면_왼쪽이_아래": True,   # 좌우가 반대로 나오면 False 로 뒤집으세요

    # 얼굴이 거의 정면이라 코-귀 신호가 약할 때만 쓰는 보조 신호 — 깊이(z).
    #   카메라(위)에 가까운 쪽이 뜬 쪽이므로, z가 큰 쪽이 바닥입니다.
    #   ※ 기준 문서는 z 사용을 금지합니다. 그건 'z로 각도를 재지 마라'는 뜻으로 보고,
    #     여기서는 각도가 아니라 좌·우 부호 판정에만 씁니다.
    #   실측 측위 72장 — 얼굴 단독 72/72 정확(보류 4장) · z 단독도 72/72 정확(보류 0장)
    #     두 신호가 어긋난 사진 0장. 얼굴이 보류된 4장을 z가 |0.32~0.49| 로 전부 구제.
    "깊이_최소차이": 0.05,
    "왼쪽z가_크면_왼쪽이_아래": True,
}

# ── 신뢰도(가림) 기준 ────────────────────────────────────────────────────────
MIN_VIS = 0.3           # 관절 하나가 이 값 미만이면 그 항목은 "측정 불가(가림)"
                        #   ※ 기준 문서 권장값은 0.5 — 이불 덮은 사진 테스트 후 결정 (현재 보류)
MIN_AVG_VIS = 0.82      # 핵심관절 평균이 이 값 미만이면 사진 전체 "판정 보류"
MIN_DETECT_CONF = 0.3   # 사람을 찾을 때 쓰는 MediaPipe 설정값 (판정 기준 아님)
# ==============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "TA_테스트결과")
CALIB_FILE = os.path.join(SCRIPT_DIR, "calib.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "pose_landmarker_lite.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


# ---------------------------------------------------------------- 관절 추출
_POSE = None


def _detect_legacy(image_bgr):
    """mediapipe 0.10.x: 내장 모델 사용, 파일 다운로드 불필요. 인스턴스는 재사용.

    반환: (관절목록, 사람실루엣 or None)
    실루엣은 히트맵 표시에만 씁니다 — 판정에 넣어봤지만 관절보다 나빴습니다.
    """
    global _POSE
    if _POSE is None:
        _POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                       enable_segmentation=True,
                                       min_detection_confidence=MIN_DETECT_CONF)
    res = _POSE.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None, None
    return res.pose_landmarks.landmark, getattr(res, "segmentation_mask", None)


def _ensure_model():
    """Tasks API용 .task 모델 확보. 없으면 1회 다운로드, 실패 시 안내하고 종료."""
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 1_000_000:
        return
    print("[안내] 포즈 모델 내려받는 중 (최초 1회, 약 6MB)...")
    import urllib.request
    tmp = MODEL_FILE + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_FILE)
    except Exception as e:                                     # 오프라인·방화벽 등
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit("[오류] 포즈 모델을 내려받지 못했습니다 (%s).\n"
                 "      인터넷이 되는 PC에서 아래 주소의 파일을 받아\n"
                 "      pose_check.py 와 같은 폴더에 두면 됩니다:\n      %s"
                 % (e, MODEL_URL))


_LANDMARKER = None


def _get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER

    from mediapipe.tasks import python as tp
    from mediapipe.tasks.python import vision
    _ensure_model()
    with open(MODEL_FILE, "rb") as f:
        blob = f.read()

    def build(base):
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base, num_poses=1,
                                         output_segmentation_masks=True,
                                         min_pose_detection_confidence=MIN_DETECT_CONF))
    try:
        _LANDMARKER = build(tp.BaseOptions(model_asset_buffer=blob))
    except Exception:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
        try:
            if not os.path.exists(tmp) or os.path.getsize(tmp) != len(blob):
                with open(tmp, "wb") as f:
                    f.write(blob)
            _LANDMARKER = build(tp.BaseOptions(model_asset_path=tmp))
        except Exception as e:
            sys.exit(f"[오류] 포즈 모델을 불러오지 못했습니다: {e}\n"
                     f"      모델 파일이 손상됐을 수 있습니다. 아래 파일을 지우고 다시 실행하세요:\n"
                     f"      {MODEL_FILE}")
    return _LANDMARKER


def _pad16(image_bgr):
    """사진 폭을 16의 배수로 맞춰 오른쪽에만 여백을 붙입니다.

    ⚠ 이걸 안 하면 세그멘테이션 마스크를 읽을 때 **파이썬 예외가 아니라 프로세스가 죽습니다.**
      mediapipe 가 마스크 각 행을 16바이트 경계에 맞춰 저장하는데, 폭×4가 16의 배수가 아니면
      메모리가 연속이 아니게 되고, numpy_view() 가 그 경우 uint8 복사 경로를 타면서
      C++ CHECK(1 == ChannelSize()) 에 걸려 abort 합니다. try/except 로 못 잡습니다.
      실측 — 폭 576(=16×36) 정상 / 폭 490·267 즉사 / 폭 236(944=16×59) 우연히 정상.
    오른쪽·아래로만 채우므로 관절의 절대 픽셀 좌표는 그대로입니다.
    """
    h, w = image_bgr.shape[:2]
    pad = (-w) % 16
    if not pad:
        return image_bgr
    return cv2.copyMakeBorder(image_bgr, 0, 0, 0, pad, cv2.BORDER_REPLICATE)


def _detect_tasks(image_bgr):
    """mediapipe 1.0+: Pose Landmarker(Tasks API). 모델 파일 필요(최초 1회 자동 다운로드)."""
    lmk = _get_landmarker()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    res = lmk.detect(mp_img)
    if not res.pose_landmarks:
        return None, None
    masks = getattr(res, "segmentation_masks", None)
    mask = None
    if masks:
        # numpy_view() 는 내부 버퍼를 그대로 보므로 복사해 둡니다(다음 프레임에 덮어씌워짐)
        mask = np.array(masks[0].numpy_view(), copy=True)
    return res.pose_landmarks[0], mask


def _pick_api(api="auto"):
    """사용할 MediaPipe 경로 결정. 0.10.x는 내장 모델(legacy), 1.0+는 Tasks API."""
    if api == "legacy" or (api == "auto" and hasattr(mp, "solutions")):
        if not hasattr(mp, "solutions"):
            sys.exit("[오류] 이 mediapipe(%s)에는 solutions API가 없습니다. --api tasks 를 쓰세요."
                     % mp.__version__)
        return _detect_legacy
    return _detect_tasks


def extract_landmarks(image_bgr, min_vis=MIN_VIS, api="auto", want_mask=False):
    """사진 1장에서 관절 좌표 추출.

    반환: (pts2d, pts3d, low_vis, avg_vis, vis)
          want_mask=True 면 뒤에 mask 를 하나 더 붙여 6개를 돌려줍니다.
      pts2d : {관절명: np.array([x, y])} 픽셀 좌표 — 판정은 전부 이것만 씁니다
      pts3d : 좌/우 판별이 애매할 때 깊이(z) 보조 신호로만 씁니다(자세판별 주석 참고).
      mask  : 사람 실루엣 0~1, 사진과 같은 크기. **히트맵 표시 전용**입니다.
              판정에 넣어 실측했지만 관절보다 나빴습니다 —
                자세 판별  관절 골반너비 1/117 오분류 vs 마스크 윤곽폭 15~46/117
                압력 예측  접촉 면적을 넣어도 R² 변화 없음(대전자 0.236→0.240, 뒤꿈치 0.213→0.187)
    """
    h0, w0 = image_bgr.shape[:2]
    padded = _pad16(image_bgr)          # 마스크 읽다 죽는 걸 막습니다 — _pad16 주석 참고
    h, w = padded.shape[:2]             # 관절 좌표는 '패딩 포함' 크기로 곱해야 원본 픽셀이 됩니다
    lms, mask = _pick_api(api)(padded)
    if lms is None:
        return None
    pts2d, pts3d, vis = {}, {}, {}
    for name, idx in POINTS.items():
        lm = lms[idx]
        pts2d[name] = np.array([lm.x * w, lm.y * h], dtype=float)
        pts3d[name] = np.array([lm.x, lm.y, getattr(lm, "z", 0.0)], dtype=float)
        vis[name] = getattr(lm, "visibility", 1.0)
    low_vis = [n for n in CORE if vis[n] < min_vis]
    avg_vis = sum(vis[n] for n in CORE) / len(CORE)
    if not want_mask:
        return pts2d, pts3d, low_vis, avg_vis, vis
    if mask is not None:
        mask = np.squeeze(np.asarray(mask, dtype=np.float32))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        mask = mask[:h0, :w0]                       # 붙였던 여백을 잘라 원본 크기로
        mask = cv2.GaussianBlur(np.clip(mask, 0.0, 1.0), (0, 0), 3.0)
    return pts2d, pts3d, low_vis, avg_vis, vis, mask


# ---------------------------------------------------------------- 기하 계산 (전부 2D)
def mid(p1, p2):
    return (p1 + p2) / 2


def axis_deg(p_from, p_to):
    """세로축(머리→발) 대비 기울기. arctan(|Δx| / |Δy|), 0~90°. 0°=세로축과 나란함."""
    d = p_to - p_from
    return math.degrees(math.atan2(abs(d[0]), abs(d[1]) + 1e-9))


def horiz_deg(p1, p2):
    """가로축(수평) 대비 기울기. arctan(|Δy| / |Δx|), 0~90°. 0°=수평."""
    d = p2 - p1
    return math.degrees(math.atan2(abs(d[1]), abs(d[0]) + 1e-9))


def interior_deg(a, b, c):
    """꼭짓점 b에서 본 a-b-c 내각 (0~180°)."""
    u, v = a - b, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def shrink_deg(now, base):
    """투영 길이가 줄어든 비율로 각도 역산. arccos(now / base), 0~90°."""
    if not base:
        return None
    r = max(0.0, min(1.0, now / base))
    return math.degrees(math.acos(r))


def dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))


# ---------------------------------------------------------------- 기준표 적용
def _band(v, green, yellow):
    """구간형 판정. green/yellow: (하한, 상한) 목록. 어디에도 안 들면 red."""
    for lo, hi in green:
        if lo <= v <= hi:
            return "green"
    for lo, hi in yellow:
        if lo <= v <= hi:
            return "yellow"
    return "red"


def _judge_value(rule, v):
    """기준표 한 항목(rule)에 값 v를 넣어 green/yellow/red 판정."""
    if "green" in rule:                                  # 구간형
        return _band(v, rule["green"], rule.get("yellow", []))
    if "green_min" in rule:                              # 최소값형
        return ("green" if v >= rule["green_min"] else
                "yellow" if v >= rule["yellow_min"] else "red")
    if "red_min" in rule:                                # 상한형(이진)
        return "red" if v >= rule["red_min"] else "green"
    raise ValueError(f"알 수 없는 기준 형식: {rule}")


def _outer_range(rule):
    """구간형 기준의 초록+노랑을 통틀어 (가장 낮은 하한, 가장 높은 상한)."""
    spans = list(rule.get("green", [])) + list(rule.get("yellow", []))
    return min(lo for lo, _ in spans), max(hi for _, hi in spans)


# 안내 문장을 사진 위 어디에 가리킬지 — 부위명 → 대표 좌표
#   화면에서는 이 점에서 사진 오른쪽 가장자리의 번호까지 지시선을 긋습니다.
부위_기준점 = {
    "머리":     lambda p: p["nose"],
    "어깨선":   lambda p: mid(p["l_shoulder"], p["r_shoulder"]),
    "골반선":   lambda p: mid(p["l_hip"], p["r_hip"]),
    "몸통":     lambda p: (mid(p["l_shoulder"], p["r_shoulder"])
                          + mid(p["l_hip"], p["r_hip"])) / 2,
    "왼다리":   lambda p: p["l_knee"],
    "오른다리": lambda p: p["r_knee"],
    "왼발":     lambda p: p.get("l_heel", p["l_ankle"]),
    "오른발":   lambda p: p.get("r_heel", p["r_ankle"]),
}


def _guide_pin(pts2d, shape, parts):
    """안내 문장이 가리킬 지점을 사진 크기 대비 0~1 비율로 돌려줍니다."""
    h, w = shape[:2]
    for p in parts:
        fn = 부위_기준점.get(p)
        if not fn:
            continue
        try:
            c = fn(pts2d)
        except KeyError:
            continue
        return {"x": round(float(c[0]) / w, 4), "y": round(float(c[1]) / h, 4)}
    return None


def _item(name, text, status, parts=(), group="자세", support=False):
    """판정 결과 한 줄.

    parts   : 결과 이미지에서 색칠할 부위 목록
    group   : "자세"   — 환자를 다시 눕혀 고치는 항목. 전체 등급을 결정합니다.
              "지지대" — 베개·발판 같은 도구가 필요하다는 신호. 전체 등급에 넣지 않고
                         표시설정["지지대_안내문구"] 한 줄로 묶어 따로 보여줍니다.
                         (등급에 섞으면 거의 모든 사진이 불량이 되어 정작 고칠 수 있는
                          자세 항목의 신호가 묻힙니다.)
    support : 자세 그룹이면서 도구 안내도 함께 필요한 항목(⑧ 족하수).
              등급에도 들어가고 supports 목록에도 올라갑니다.
    """
    return {"name": name, "text": text, "status": status, "parts": list(parts),
            "group": group, "support": support}


def _ok(vis, *names, **kw):
    """지정한 관절들이 전부 최소 신뢰도를 넘는지."""
    m = kw.get("min_vis", MIN_VIS)
    return all(vis.get(n, 0.0) >= m for n in names)


# ---------------------------------------------------------------- 촬영 방향 · 캘리브레이션
def check_orientation(pts2d):
    """머리가 위쪽으로 찍혔는지 확인. 어긋나면 안내 문구를 반환, 정상이면 None."""
    sh = mid(pts2d["l_shoulder"], pts2d["r_shoulder"])
    hp = mid(pts2d["l_hip"], pts2d["r_hip"])
    if sh[1] >= hp[1]:
        return "촬영 방향 확인 — 머리가 사진 위쪽에 오도록 찍어 주세요"
    return None


def measure_calibration(pts2d):
    """앙와위·침대 평평 상태에서 1회 측정하는 기준값.

    W0 = 어깨너비(11-12)             → 상체 회전각 역산 · 골반이 가렸을 때 폴백
    H0 = 골반너비(23-24)             → 하체 회전각 역산 (자세 판별의 주 신호)
    L0 = 몸통길이(어깨중점~골반중점) → 상체 기울기(침대머리) 역산에 사용
    """
    return {
        "W0": dist(pts2d["l_shoulder"], pts2d["r_shoulder"]),
        "H0": dist(pts2d["l_hip"], pts2d["r_hip"]),
        "L0": dist(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
                   mid(pts2d["l_hip"], pts2d["r_hip"])),
    }


def _calib_get(calib, key):
    """캘리브레이션 값 꺼내기. 없거나 0이면 None."""
    if not calib:
        return None
    try:
        v = float(calib.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# ---------------------------------------------------------------- 자세 판별
def detect_pose_type(pts2d, vis=None, calib=None):
    """앙와위인지 측위인지 판별. 골반이 주 신호, 어깨는 골반이 가렸을 때의 폴백입니다.

    왜 골반인가 — 자세판별 상단 주석 참고. 어깨는 팔 자세에 그대로 흔들립니다
    (앙와위 회전각 중앙값이 어깨 18.1° vs 골반 4.8°).

    반환: (pose_type, 대표회전각 or None, 근사여부, 상세)
      pose_type  : "앙와위" / "측위"
      대표회전각 : 기준값이 있을 때만 숫자. 없으면 None.
      근사여부   : True면 기준값 없이 몸통길이 대비 비율로 추정한 것
      상세       : {"기준", "어깨각", "골반각", "상체각", "하체각", "비틀림", "비틀림_근사"}
                   어깨각·골반각 : W0·H0가 있어야 채워집니다(② 측위 각도가 이 값을 씁니다).
                   상체각·하체각 : 비틀림 계산에 실제로 쓴 값. 기준값이 없으면 인구
                                   중앙값으로 근사해서라도 채웁니다.
    """
    sh_w = dist(pts2d["l_shoulder"], pts2d["r_shoulder"])
    hp_w = dist(pts2d["l_hip"], pts2d["r_hip"])
    trunk = dist(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
                 mid(pts2d["l_hip"], pts2d["r_hip"])) + 1e-9

    어깨각 = shrink_deg(sh_w, _calib_get(calib, "W0"))
    골반각 = shrink_deg(hp_w, _calib_get(calib, "H0"))

    # 비틀림은 기준값이 없어도 봐야 합니다. 없으면 앙와위 인구 중앙값으로 각도를 근사합니다.
    # (안 그러면 캘리브레이션 전에는 "상체 앙와위 + 하체 측위"가 그냥 측위로 넘어갑니다.)
    근사 = 어깨각 is None or 골반각 is None
    상체각 = 어깨각 if not 근사 else shrink_deg(sh_w / trunk, 자세판별["앙와위_어깨몸통비"])
    하체각 = 골반각 if not 근사 else shrink_deg(hp_w / trunk, 자세판별["앙와위_골반몸통비"])
    상세 = {"기준": None, "어깨각": 어깨각, "골반각": 골반각,
            "상체각": 상체각, "하체각": 하체각, "비틀림_근사": 근사,
            "비틀림": (abs(상체각 - 하체각) if 상체각 is not None and 하체각 is not None
                     else None)}

    골반보임 = vis is None or _ok(vis, "l_hip", "r_hip")
    if 골반보임:
        if 골반각 is not None:
            상세["기준"] = "골반"
            return ("측위" if 골반각 >= 자세판별["측위_최소각도"] else "앙와위"), 골반각, False, 상세
        상세["기준"] = "골반(근사)"
        kind = "측위" if hp_w / trunk < 자세판별["근사_골반몸통비"] else "앙와위"
        return kind, None, True, 상세

    # 골반이 이불 등에 가림 — 정확도가 떨어지는 어깨로 폴백
    if 어깨각 is not None:
        상세["기준"] = "어깨"
        return ("측위" if 어깨각 >= 자세판별["측위_최소각도"] else "앙와위"), 어깨각, False, 상세
    상세["기준"] = "어깨(근사)"
    kind = "측위" if sh_w / trunk < 자세판별["근사_어깨몸통비"] else "앙와위"
    return kind, None, True, 상세


def detect_down_side(pts2d, vis, pts3d=None):
    """측위에서 어느 쪽 어깨가 바닥에 닿았는지(=아래쪽 다리) 판별.

    옆으로 누우면 얼굴이 위로 올라온 쪽을 향하므로, 코가 귀중점에서 좌우 어느 쪽으로
    벗어났는지로 알 수 있습니다.
      실측 10장 — 값이 -0.16~-0.08 과 +0.13 두 무리로 뚜렷하게 갈림
      정답 3장 대조 — 3/3 일치
    ※ MediaPipe visibility 로는 판별할 수 없습니다. 좌우 차이가 0.000~0.005로 잡음 수준입니다.

    반환: "l"(왼쪽 어깨가 바닥) / "r"(오른쪽 어깨가 바닥) / None(판별 보류)
          l·r 은 MediaPipe 기준 = 환자 본인 기준 좌·우입니다.
    """
    if not _ok(vis, "l_ear", "r_ear", "nose"):
        return None
    ear_mid = mid(pts2d["l_ear"], pts2d["r_ear"])
    trunk = dist(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
                 mid(pts2d["l_hip"], pts2d["r_hip"])) + 1e-9
    s = (pts2d["nose"][0] - ear_mid[0]) / trunk
    if abs(s) >= 자세판별["좌우판별_최소차이"]:
        return "l" if (s > 0) == 자세판별["코가_화면오른쪽이면_왼쪽이_아래"] else "r"

    # 얼굴이 거의 정면 — 깊이(z)로 한 번 더 봅니다. 없으면 보류.
    if pts3d is not None and _ok(vis, "l_shoulder", "r_shoulder", "l_hip", "r_hip"):
        dz = ((pts3d["l_shoulder"][2] - pts3d["r_shoulder"][2])
              + (pts3d["l_hip"][2] - pts3d["r_hip"][2])) / 2
        if abs(dz) >= 자세판별["깊이_최소차이"]:
            return "l" if (dz > 0) == 자세판별["왼쪽z가_크면_왼쪽이_아래"] else "r"
    return None                                        # 좌우 판별 보류


# ---------------------------------------------------------------- 항목별 판정 · 앙와위
def _sup_trunk_align(pts2d):
    v = axis_deg(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
                 mid(pts2d["l_hip"], pts2d["r_hip"]))
    return _item("① 몸통 정렬", f"세로축에서 {v:.0f}° 틀어짐",
                 _judge_value(기준_앙와위["① 몸통 정렬"], v), ["몸통"])

def _sup_head_shift(pts2d, vis):
    if not _ok(vis, "nose"):
        return _item("② 머리 쏠림", "측정 불가 (가려짐)", "info", ["머리"])
    v = axis_deg(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]), pts2d["nose"])
    return _item("② 머리 쏠림", f"중심에서 {v:.0f}° 벗어남",
                 _judge_value(기준_앙와위["② 머리 쏠림"], v), ["머리"])

def _sup_shoulder_level(pts2d):
    v = horiz_deg(pts2d["l_shoulder"], pts2d["r_shoulder"])
    return _item("③ 어깨 수평", f"수평에서 {v:.0f}° 기울어짐",
                 _judge_value(기준_앙와위["③ 어깨 수평"], v), ["어깨선"])

def _sup_hip_level(pts2d):
    v = horiz_deg(pts2d["l_hip"], pts2d["r_hip"])
    return _item("④ 골반 수평", f"수평에서 {v:.0f}° 기울어짐",
                 _judge_value(기준_앙와위["④ 골반 수평"], v), ["골반선"])

def _sup_head_tilt(pts2d, vis):
    if not _ok(vis, "l_eye", "r_eye"):
        return _item("⑤ 머리 갸우뚱", "측정 불가 (고개 돌아감)", "info", ["머리"])
    v = horiz_deg(pts2d["l_eye"], pts2d["r_eye"])
    return _item("⑤ 머리 갸우뚱", f"고개가 {v:.0f}° 기울어짐",
                 _judge_value(기준_앙와위["⑤ 머리 갸우뚱"], v), ["머리"])

def _sup_hob(pts2d, calib):
    l0 = _calib_get(calib, "L0")
    if not l0:
        return _item("⑥ 상체 기울기", "측정 불가 (앙와위 기준값이 없습니다)", "info", ["몸통"])
    now = dist(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
               mid(pts2d["l_hip"], pts2d["r_hip"]))
    v = shrink_deg(now, l0)
    note = ""
    w0 = _calib_get(calib, "W0")
    if w0:
        roll = shrink_deg(dist(pts2d["l_shoulder"], pts2d["r_shoulder"]), w0)
        if roll >= 자세판별["상체기울기_경고_회전각"]:
            note = f" ⚠ 몸이 {roll:.0f}° 돌아가 있어 각도가 부정확할 수 있습니다"
    return _item("⑥ 상체 기울기", f"침대 머리 각도 {v:.0f}°{note}",
                 _judge_value(기준_앙와위["⑥ 상체 기울기"], v), ["몸통"])

def _knee_extension(pts2d, vis, rule, name, sides=("l", "r")):
    vals, parts = {}, []
    for s in sides:
        if _ok(vis, f"{s}_hip", f"{s}_knee", f"{s}_ankle"):
            vals[s] = interior_deg(pts2d[f"{s}_hip"], pts2d[f"{s}_knee"], pts2d[f"{s}_ankle"])
            parts.append("왼다리" if s == "l" else "오른다리")
    if not vals:
        return _item(name, "측정 불가 (가려짐)", "info", parts, group="지지대")
    worst = max(vals.values())
    txt = " / ".join(f"{'왼쪽' if s == 'l' else '오른쪽'} {v:.0f}°" for s, v in vals.items())
    return _item(name, f"{txt} (관절 펴짐 정도)", _judge_value(rule, worst), parts, group="지지대")

def _sup_leg_gap(pts2d, vis):
    if not _ok(vis, "l_ankle", "r_ankle"):
        return _item("⑩ 다리 모임", "측정 불가 (가려짐)", "info", ["왼다리", "오른다리"])
    hip_w = dist(pts2d["l_hip"], pts2d["r_hip"]) + 1e-9
    r = dist(pts2d["l_ankle"], pts2d["r_ankle"]) / hip_w
    return _item("⑩ 다리 모임", f"발목 간격 비율 {r:.2f}",
                 _judge_value(기준_앙와위["⑩ 다리 모임"], r), ["왼다리", "오른다리"])


# ---------------------------------------------------------------- 항목별 판정 · 측위
def _lat_trunk_line(pts2d, vis):
    if _ok(vis, "l_ear", "r_ear"):
        head = mid(pts2d["l_ear"], pts2d["r_ear"])
    elif _ok(vis, "nose"):
        head = pts2d["nose"]
    else:
        return _item("① 몸통 일직선", "측정 불가 (가려짐)", "info", ["몸통"])
    sh = mid(pts2d["l_shoulder"], pts2d["r_shoulder"])
    hp = mid(pts2d["l_hip"], pts2d["r_hip"])
    v = abs(180 - interior_deg(head, sh, hp))
    return _item("① 몸통 일직선", f"{v:.0f}° 꺾임",
                 _judge_value(기준_측위["① 몸통 일직선"], v), ["몸통", "머리"])


def _lat_angle(pts2d, calib, roll=None):
    roll = roll or {}
    골반각, 어깨각 = roll.get("골반각"), roll.get("어깨각")
    v, 기준 = (골반각, "골반") if 골반각 is not None else (어깨각, "어깨")
    if v is None:
        return _item("② 측위 각도", "측정 불가 (앙와위 기준값이 없습니다)", "info", ["어깨선"])
    rule = 기준_측위["② 측위 각도"]
    lo, hi = _outer_range(rule)
    note = "" if lo <= v <= hi else (" (앙와위에 가까움)" if v < lo else " (90° 접근)")
    both = f" (상체 {어깨각:.0f}° / 하체 {골반각:.0f}°)" if 골반각 is not None and 어깨각 is not None else ""
    return _item("② 측위 각도", f"돌아누운 각도 {v:.0f}° ({기준} 기준){note}{both}",
                 _judge_value(rule, v), ["어깨선"])


def _lat_leg_gap(pts2d, vis):
    judged = 표시설정["측위_다리모임_판정함"]
    trunk = dist(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
                 mid(pts2d["l_hip"], pts2d["r_hip"])) + 1e-9
    got, states = [], []
    for key, a, b in [("⑥ 다리 모임(발목)", "l_ankle", "r_ankle"),
                      ("⑥ 다리 모임(무릎)", "l_knee", "r_knee")]:
        if not _ok(vis, a, b):
            continue
        r = dist(pts2d[a], pts2d[b]) / trunk
        got.append(f"{'발목' if 'ankle' in a else '무릎'} {r:.2f}")
        states.append(_judge_value(기준_측위[key], r))
    if not states:
        return _item("⑥ 다리 모임", "측정 불가 (가려짐)", "info", ["왼다리", "오른다리"])
    txt = " / ".join(got)
    if not judged:
        return _item("⑥ 다리 모임", f"간격 비율: {txt} (판정 기준 미정)", "info", ["왼다리", "오른다리"])
    return _item("⑥ 다리 모임", f"간격 비율: {txt}", max(states, key=lambda s: RANK[s]),
                 ["왼다리", "오른다리"])


# ---------------------------------------------------------------- 항목별 판정 · 공통
def _legs_crossed(pts2d, vis):
    normal = 기준_공통["⑨ 다리 교차"]["정상은_왼발목이_더_오른쪽"]
    bad, checked = [], []
    for lab, a, b in [("발목", "l_ankle", "r_ankle"), ("무릎", "l_knee", "r_knee")]:
        if not _ok(vis, a, b):
            continue
        checked.append(lab)
        if (pts2d[a][0] > pts2d[b][0]) != normal:
            bad.append(lab)
    if not checked:
        return _item("⑨ 다리 교차", "측정 불가 (가려짐)", "info", ["왼다리", "오른다리"])
    txt = f"{', '.join(bad)} 좌우 역전됨" if bad else f"{', '.join(checked)} 정상"
    return _item("⑨ 다리 교차", txt, "red" if bad else "green",
                 ["왼다리", "오른다리"])


def _foot_drop(pts2d, vis):
    rule = 기준_공통["⑧ 족하수"]
    R = rule["발길이_몸통비"]
    sh = mid(pts2d["l_shoulder"], pts2d["r_shoulder"])
    hp = mid(pts2d["l_hip"], pts2d["r_hip"])
    trunk = dist(sh, hp) + 1e-9
    axis = (hp - sh) / trunk

    angs, txts = [], []
    for s, lab in [("l", "왼쪽"), ("r", "오른쪽")]:
        if not _ok(vis, f"{s}_heel", f"{s}_foot"):
            continue
        proj = float(np.dot(pts2d[f"{s}_foot"] - pts2d[f"{s}_heel"], axis)) / trunk
        if proj <= 0:
            a = 90.0
        else:
            a = math.degrees(math.acos(min(1.0, proj / R)))
        angs.append(a)
        txts.append(f"{lab} {a:.0f}°")
    if not angs:
        return _item("⑧ 족하수", "측정 불가 (가려짐)", "info", ["왼발", "오른발"], group="지지대")
    worst = min(angs)
    return _item("⑧ 족하수", " / ".join(txts) + " (90°가 정상)",
                 _judge_value(rule, worst), ["왼발", "오른발"], group="지지대")


def _foot_dir_ref(pts2d, vis):
    axis = (mid(pts2d["l_hip"], pts2d["r_hip"])
            - mid(pts2d["l_shoulder"], pts2d["r_shoulder"]))
    hip_mid = mid(pts2d["l_hip"], pts2d["r_hip"])
    vals = []
    for s, lab in [("l", "왼쪽"), ("r", "오른쪽")]:
        if not _ok(vis, f"{s}_heel", f"{s}_foot", f"{s}_hip"):
            continue
        v = pts2d[f"{s}_foot"] - pts2d[f"{s}_heel"]
        cos = np.dot(axis, v) / (np.linalg.norm(axis) * np.linalg.norm(v) + 1e-9)
        dev = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        side = "바깥" if np.dot(v, pts2d[f"{s}_hip"] - hip_mid) > 0 else "안쪽"
        vals.append(f"{lab} {side} {dev:.0f}°")
    txt = " / ".join(vals) if vals else "측정 불가 (가려짐)"
    return _item("발끝 방향", txt + " (0°가 바른 방향)", "info", ["왼발", "오른발"])


def _elbow_ref(pts2d, vis):
    vals = []
    for s, lab in [("l", "왼쪽"), ("r", "오른쪽")]:
        if _ok(vis, f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"):
            a = interior_deg(pts2d[f"{s}_shoulder"], pts2d[f"{s}_elbow"],
                             pts2d[f"{s}_wrist"])
            vals.append(f"{lab} {a:.0f}°")
    txt = " / ".join(vals) if vals else "측정 불가 (가려짐)"
    return _item("팔꿈치 굴곡", txt + " (목표 각도는 상황마다 다름)", "info", ["왼팔", "오른팔"])


# ---------------------------------------------------------------- 종합
def _summarize(items):
    """항목 목록 → (전체 상태, 요약문, 표시줄, 부위별 색, 지지대 안내)."""
    posture = [it for it in items if it["group"] == "자세" and it["status"] in RANK]
    worst = max((it["status"] for it in posture), key=lambda s: RANK[s], default="green")
    bad = [it["name"] for it in posture if it["status"] == "red"]
    warn = [it["name"] for it in posture if it["status"] == "yellow"]
    if worst == "green":
        verdict = "자세 정상"
    elif worst == "yellow":
        verdict = f"주의 {len(warn)}건 — {', '.join(warn)}"
    else:
        verdict = f"불량 — {', '.join(bad)}"

    flagged = [it for it in items
               if it.get("support")
               or (it["group"] == "지지대" and it["status"] in ("yellow", "red"))]
    문구 = 표시설정["지지대_문구"]
    support = {
        "needed": bool(flagged),
        "message": 표시설정["지지대_안내문구"] if flagged else "",
        "items": [it["name"] for it in flagged],
        "details": [{"item": it["name"],
                     "do": 문구.get(it["name"], {}).get("do", ""),
                     "why": 문구.get(it["name"], {}).get("why", ""),
                     "status": it["status"]}
                    for it in flagged],
        "status": (max((it["status"] for it in flagged if it["status"] in RANK),
                       key=lambda s: RANK[s], default="green") if flagged else "green"),
    }

    # ⭐️ 텍스트 출력 형식을 부드럽게 수정
    lines = [f"{it['name']}: {it['text']} [{LABEL[it['status']]}]" for it in items]

    문구 = 표시설정["자세_문구"]

    def _key(it):
        if it["name"] == "② 측위 각도" and "앙와위에 가까움" in it["text"]:
            return "② 측위 각도(얕음)"
        return it["name"]

    def _rank(it):
        return 문구.get(_key(it), {}).get("순위", 99)

    문제 = sorted([it for it in posture if it["status"] in ("yellow", "red")],
                 key=lambda it: (-RANK[it["status"]], _rank(it)))
    guides = [{"item": it["name"], "status": it["status"],
               "parts": it["parts"], **{k: v for k, v in 문구.get(_key(it), {}).items()
                                        if k != "순위"}}
              for it in 문제]
    good = [표시설정["자세_잘됨"][it["name"]]
            for it in sorted([x for x in posture if x["status"] == "green"], key=_rank)
            if it["name"] in 표시설정["자세_잘됨"]]

    parts = {}
    for it in items:
        if it["status"] not in RANK:
            continue
        for p in it["parts"]:
            if p not in parts or RANK[it["status"]] > RANK[parts[p]]:
                parts[p] = it["status"]
    return worst, verdict, lines, parts, support, guides, good


def judge(pts2d, vis, calib=None, pose_type="auto", pts3d=None, shape=None):
    """자세 판별 + 항목별 판정. 결과를 딕셔너리로 돌려줍니다."""
    warning = check_orientation(pts2d)

    auto_kind, lat_angle, approx, roll = detect_pose_type(pts2d, vis, calib)
    kind = pose_type if pose_type in ("앙와위", "측위") else auto_kind

    twist = roll["비틀림"]
    twist_limit = 자세판별["비틀림_보류각_근사" if roll["비틀림_근사"] else "비틀림_보류각"]
    twist_hold = twist is not None and twist >= twist_limit

    down = None                          
    if kind == "앙와위":
        items = [
            _sup_trunk_align(pts2d),
            _sup_head_shift(pts2d, vis),
            _sup_shoulder_level(pts2d),
            _sup_hip_level(pts2d),
            _sup_head_tilt(pts2d, vis),
            _sup_hob(pts2d, calib),
            _knee_extension(pts2d, vis, 기준_앙와위["⑦ 무릎 신전"], "⑦ 무릎 신전"),
            _foot_drop(pts2d, vis),
            _legs_crossed(pts2d, vis),
            _sup_leg_gap(pts2d, vis),
            _foot_dir_ref(pts2d, vis),
            _elbow_ref(pts2d, vis),
        ]
    else:
        down = detect_down_side(pts2d, vis, pts3d)   
        knee_name = ("③ 무릎 신전 (아래쪽 불명 · 좌우 모두)" if down is None
                     else f"③ 무릎 신전 ({'왼쪽' if down == 'l' else '오른쪽'} 어깨 바닥)")
        items = [
            _lat_trunk_line(pts2d, vis),
            _lat_angle(pts2d, calib, roll),
            _knee_extension(pts2d, vis, 기준_측위["③ 무릎 신전"], knee_name,
                            sides=("l", "r") if down is None else (down,)),
            _foot_drop(pts2d, vis),
            _lat_leg_gap(pts2d, vis),
            _foot_dir_ref(pts2d, vis),
            _elbow_ref(pts2d, vis),
        ]

    status, verdict, lines, parts, support, guides, good = _summarize(items)
    
    if shape is not None:
        for g in guides:
            g["pin"] = _guide_pin(pts2d, shape, g["parts"])
            
    # ⭐️ 앙와위 기준값이 없어 각도를 모를 때 읽기 편한 문구로 추가
    if approx and kind == "측위":
        lines.insert(0, "💡 앙와위(똑바로 누움) 기준값이 없어 구체적인 각도 결과를 제공할 수 없습니다.")
        
    if twist_hold:
        lines.insert(0, f"⚠ 몸통 비틀림 {twist:.0f}°{'(근사)' if roll['비틀림_근사'] else ''} "
                        f"(상체 {roll['상체각']:.0f}° · 하체 {roll['하체각']:.0f}°) — 판정 보류")
    return {
        "pose_type": kind, "status": status, "verdict": verdict, "lines": lines,
        "items": items, "parts": parts, "warning": warning, "approx": approx,
        "lat_angle": lat_angle, "support": support,
        "roll": roll, "twist": twist, "twist_hold": twist_hold,
        "guides": guides,          
        "good_points": good,       
        "down_side": down,
    }


def judge_top_view(pts2d, pts3d=None, vis=None, calib=None):
    """하위호환용 — 기존 호출 형태 (status, verdict, lines) 그대로 돌려줍니다."""
    r = judge(pts2d, vis, calib=calib, pts3d=pts3d)
    return r["status"], r["verdict"], r["lines"]


# ---------------------------------------------------------------- 예상 압력 히트맵
압력예상 = {
    "앙와위":     {"뒤통수": 2.5, "어깨": 0.8, "천골": 2.3, "대전자": 1.8,
                  "무릎": 0.65, "뒤꿈치": 3.4},
    "측위_얕음":  {"뒤통수": 1.8, "어깨": 2.0, "대전자": 2.9, "무릎": 2.2, "뒤꿈치": 0.5},
    "측위_보통":  {"뒤통수": 1.6, "어깨": 2.2, "대전자": 4.7, "무릎": 2.1, "뒤꿈치": 1.1},
    "측위_깊음":  {"뒤통수": 1.5, "어깨": 2.8, "대전자": 5.0, "무릎": 2.6, "뒤꿈치": 0.8},
}
압력표시 = {
    "구간경계": (45, 60),     
    "최대배수": 5.0,          
    "번짐": 0.12,             
    "몸바닥값": 0.75,         
    "진하기": 0.85,           
    "최소진하기": 0.42,       
    "탈색": 0.85,             
}


def _pressure_lut():
    anchors = [(0.00, (95, 185, 70)),     # 초록
               (0.40, (70, 225, 235)),    # 노랑
               (0.70, (40, 150, 250)),    # 주황
               (1.00, (48, 40, 205))]     # 빨강  (모두 BGR)
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(anchors[:-1], anchors[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0 + 1e-9)
                lut[i] = [int(c0[j] + (c1[j] - c0[j]) * k) for j in range(3)]
                break
    return lut


def _body_mask(pts2d, vis, shape, trunk):
    m = np.zeros(shape[:2], np.uint8)
    P = lambda k: (int(pts2d[k][0]), int(pts2d[k][1]))
    torso = [k for k in ("l_shoulder", "r_shoulder", "r_hip", "l_hip") if _ok(vis, k)]
    if len(torso) == 4:
        cv2.fillConvexPoly(m, np.array([P(k) for k in torso], np.int32), 255)
    limbs = [("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
             ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
             ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_heel"),
             ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_heel")]
    t = max(5, int(trunk * 0.12))
    for a, b in limbs:
        if _ok(vis, a, b):
            cv2.line(m, P(a), P(b), 255, t)
    if _ok(vis, "l_ear", "r_ear"):
        c = mid(pts2d["l_ear"], pts2d["r_ear"])
        cv2.circle(m, (int(c[0]), int(c[1])), max(7, int(trunk * 0.15)), 255, -1)
    return cv2.GaussianBlur(m, (0, 0), max(1.5, trunk * 0.022)) / 255.0


_PRESSURE_LUT = _pressure_lut()


def _pressure_sites(pts2d, vis, kind, roll, down):
    lo, hi = 압력표시["구간경계"]
    if kind == "앙와위":
        key = "앙와위"
    else:
        key = ("측위_얕음" if roll is None or roll < lo else
               "측위_보통" if roll < hi else "측위_깊음")
    v = 압력예상[key]
    out = []

    def add(name, pt_keys, val):
        if all(_ok(vis, k) for k in pt_keys):
            pts = [pts2d[k] for k in pt_keys]
            out.append((sum(pts) / len(pts), val, name))

    if _ok(vis, "l_ear", "r_ear", "l_shoulder", "r_shoulder"):
        ear = mid(pts2d["l_ear"], pts2d["r_ear"])
        sh = mid(pts2d["l_shoulder"], pts2d["r_shoulder"])
        out.append((ear + (sh - ear) * 0.25, v["뒤통수"], "뒤통수"))

    if kind == "앙와위":
        add("천골", ["l_hip", "r_hip"], v["천골"])
        for s in ("l", "r"):
            add("어깨", [f"{s}_shoulder"], v["어깨"])
            add("대전자", [f"{s}_hip"], v["대전자"])
            add("무릎", [f"{s}_knee"], v["무릎"])
            add("뒤꿈치", [f"{s}_heel"], v["뒤꿈치"])
    else:
        s = down or "l"          
        add("어깨", [f"{s}_shoulder"], v["어깨"])
        add("대전자", [f"{s}_hip"], v["대전자"])
        add("무릎", [f"{s}_knee"], v["무릎"])
        add("복사뼈", [f"{s}_ankle"], v["뒤꿈치"])
    return out, key


def draw_pressure(img, pts2d, vis, kind, roll=None, down=None, mask=None):
    sites, key = _pressure_sites(pts2d, vis, kind, roll, down)
    h, w = img.shape[:2]
    if not sites:
        return img.copy(), key
    trunk = dist(mid(pts2d["l_shoulder"], pts2d["r_shoulder"]),
                 mid(pts2d["l_hip"], pts2d["r_hip"])) + 1e-9

    body = mask if mask is not None else _body_mask(pts2d, vis, img.shape, trunk)
    heat = body * 압력표시["몸바닥값"]

    sig = max(6.0, trunk * 압력표시["번짐"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for c, val, _ in sites:
        d2 = (xx - float(c[0])) ** 2 + (yy - float(c[1])) ** 2
        np.maximum(heat, val * np.exp(-d2 / (2 * sig * sig)), out=heat)

    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    g = (body * 압력표시["탈색"])[..., None]
    base = img * (1 - g) + gray * g

    n = np.clip(heat / 압력표시["최대배수"], 0, 1)
    color = _PRESSURE_LUT[(n * 255).astype(np.uint8)]
    lo, hi = 압력표시["최소진하기"], 압력표시["진하기"]
    strength = np.clip(np.maximum(body, heat / 압력표시["몸바닥값"]), 0, 1)
    a = (strength * (lo + (hi - lo) * n ** 0.7))[..., None]
    return (base * (1 - a) + color * a).astype(np.uint8), key


# ---------------------------------------------------------------- 안내 번호 레일
레일표시 = {
    "레일폭비": 0.16,        
    "최소레일폭": 54,        
    "배지비": 0.66,          
    "위여백비": 0.06,        
    "간격비": 0.30,          
    "점선": (9, 6),          
}
_배지색 = {"red": (72, 73, 227), "yellow": (0, 161, 237), "green": (80, 200, 80)}


def _dashed(img, p1, p2, color, thick=2, pattern=(9, 6)):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    total = float(np.linalg.norm(p2 - p1))
    if total < 1e-6:
        return
    unit = (p2 - p1) / total
    on, off = pattern
    d = 0.0
    while d < total:
        a = p1 + unit * d
        b = p1 + unit * min(d + on, total)
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 color, thick, cv2.LINE_AA)
        d += on + off


def draw_guide_rail(img, guides, max_n=None):
    if max_n is None:
        max_n = 표시설정["안내_최대개수"]
    guides = [g for g in (guides or [])][:max_n]
    if not guides:
        return img

    h, w = img.shape[:2]
    rail = max(레일표시["최소레일폭"], int(round(w * 레일표시["레일폭비"])))
    d = int(round(rail * 레일표시["배지비"]))          
    r = d // 2
    gap = int(round(d * 레일표시["간격비"]))
    top = int(round(h * 레일표시["위여백비"])) + r

    need = top + (d + gap) * (len(guides) - 1) + r
    if need > h - r:
        gap = max(4, (h - 2 * r - top - r) // max(1, len(guides) - 1))

    out = np.full((h, w + rail, 3), 255, np.uint8)
    out[:, :w] = img
    cx = w + rail // 2

    for i, g in enumerate(guides):
        cy = top + (d + gap) * i
        col = _배지색.get(g.get("status"), _배지색["yellow"])
        pin = g.get("pin")
        if pin:
            px = int(round(float(pin["x"]) * w))
            py = int(round(float(pin["y"]) * h))
            _dashed(out, (cx - r - 2, cy), (px, py), (170, 170, 165), 2, 레일표시["점선"])
            cv2.circle(out, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), 5, col, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, col, -1, cv2.LINE_AA)
        txt = str(i + 1)
        scale = d / 34.0
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        cv2.putText(out, txt, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- 오버레이
SKELETON = [
    ("l_shoulder", "r_shoulder", "어깨선"),
    ("l_hip", "r_hip", "골반선"),
    ("l_shoulder", "l_hip", "몸통"), ("r_shoulder", "r_hip", "몸통"),
    ("_sh_mid", "_hip_mid", "몸통"),      
    ("nose", "_sh_mid", "머리"),          
    ("l_eye", "r_eye", "머리"), ("nose", "l_eye", "머리"), ("nose", "r_eye", "머리"),
    ("l_ear", "r_ear", "머리"),           
    ("l_shoulder", "l_elbow", "왼팔"), ("l_elbow", "l_wrist", "왼팔"),
    ("r_shoulder", "r_elbow", "오른팔"), ("r_elbow", "r_wrist", "오른팔"),
    ("l_hip", "l_knee", "왼다리"), ("l_knee", "l_ankle", "왼다리"),
    ("r_hip", "r_knee", "오른다리"), ("r_knee", "r_ankle", "오른다리"),
    ("l_ankle", "l_heel", "왼발"), ("l_heel", "l_foot", "왼발"),
    ("r_ankle", "r_heel", "오른발"), ("r_heel", "r_foot", "오른발"),
]

_RANK_BY_COLOR = {GREEN: 0, GRAY: 0, YELLOW: 1, RED: 2}


def draw_overlay(img, pts, status="green", verdict="", lines=(), hold=False, parts=None):
    t = max(2, img.shape[1] // 400)
    base = YELLOW if hold else COLOR.get(status, GREEN)

    P = dict(pts)
    if "l_shoulder" in P and "r_shoulder" in P:
        P["_sh_mid"] = mid(P["l_shoulder"], P["r_shoulder"])
    if "l_hip" in P and "r_hip" in P:
        P["_hip_mid"] = mid(P["l_hip"], P["r_hip"])

    def edge_color(part):
        if hold or parts is None:
            return base
        return COLOR.get(parts.get(part, "green"), GREEN)

    joint = {}
    for a, b, part in SKELETON:
        if a not in P or b not in P:
            continue
        c = edge_color(part)
        cv2.line(img, tuple(P[a].astype(int)), tuple(P[b].astype(int)), c, t)
        for n in (a, b):
            prev = joint.get(n)
            if prev is None or _RANK_BY_COLOR.get(c, 0) > _RANK_BY_COLOR.get(prev, 0):
                joint[n] = c

    for n, p in pts.items():
        c = joint.get(n, base)
        cv2.circle(img, tuple(p.astype(int)), t * 3, WHITE, -1)
        cv2.circle(img, tuple(p.astype(int)), t * 3, c, t)
    return img


# ---------------------------------------------------------------- 한 장 처리
EXIT_CODE = {"green": 0, "yellow": 4, "red": 2, "hold": 3, "error": 1}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_calib(path=CALIB_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_calib(calib, path=CALIB_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)


def check_image(path, out=None, calib=None, pose_type="auto", min_avg_vis=MIN_AVG_VIS,
                api="auto", quiet=False, open_result=False):
    def fail(msg):
        if not quiet:
            print(f"[오류] {msg}")
        return {"path": path, "tag": "오류", "verdict": msg, "lines": [],
                "avg_vis": None, "out": None, "pose_type": "", "code": EXIT_CODE["error"]}

    img = imread_u(path)
    if img is None:
        return fail(f"사진을 열 수 없습니다(파일 없음 또는 지원하지 않는 형식): {path}")

    found = extract_landmarks(img, api=api)
    if found is None:
        return fail("사람을 찾지 못했습니다. 몸 전체가 나오게 다시 촬영해 주세요.")
    pts, pts3d, low_vis, avg_vis, vis = found

    r = judge(pts, vis, calib=calib, pose_type=pose_type, pts3d=pts3d)
    status, verdict, lines, parts = r["status"], r["verdict"], r["lines"], r["parts"]

    hold = avg_vis < min_avg_vis or r["twist_hold"]
    if avg_vis < min_avg_vis:
        verdict = f"가림 감지 (신뢰도 {avg_vis:.0%}) — 판정 보류"
        lines = [ln + " ← 참고용" for ln in lines]
        lines.append("이불을 걷거나 몸이 보이게 다시 촬영해 주세요")
    elif r["twist_hold"]:
        verdict = f"몸통 비틀림 {r['twist']:.0f}° — 판정 보류"
        lines = [ln + " ← 참고용" for ln in lines]
        lines.append("상체와 하체가 같은 쪽을 보도록 다시 눕혀 주세요")
    if r["warning"]:
        lines.insert(0, "⚠ " + r["warning"])

    tag = "보류" if hold else LABEL[status]
    if not quiet:
        print(f"[{tag}] ({r['pose_type']}) {verdict}")
        for ln in lines:
            print("  - " + ln)
        if r["support"]["needed"]:
            print(f"  ▸ {r['support']['message']}  ({', '.join(r['support']['items'])})")

    out = out or os.path.join(DEFAULT_OUT_DIR, auto_name(path))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if not imwrite_u(out, draw_overlay(img, pts, status, verdict, lines,
                                       hold=hold, parts=parts)):
        return fail(f"결과 이미지를 저장할 수 없습니다: {out}")
    if not quiet:
        print(f"결과 저장: {out}")
    if open_result and hasattr(os, "startfile"):
        try:
            os.startfile(os.path.abspath(out))
        except OSError:
            pass
    return {"path": path, "tag": tag, "verdict": verdict, "lines": lines,
            "avg_vis": avg_vis, "out": out, "pose_type": r["pose_type"],
            "support": r["support"],
            "code": EXIT_CODE["hold"] if hold else EXIT_CODE[status]}


def auto_name(path):
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    tags = []
    if len(parts) >= 4 and parts[-3].upper() in ("RGB", "IR", "DEPTH", "PM", "IRRAW", "DEPTHRAW"):
        tags = [parts[-2], parts[-4]]
    elif len(parts) >= 2:
        tags = [parts[-2]]
    return "_".join([t for t in tags if t] + [stem]) + "_result.jpg"


def _collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.lower().endswith(IMG_EXT) and "_result" not in n.lower()]
        else:
            files.append(p)
    return files


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(
        description="누운 자세 판정 — 엑셀 기준표(position_criteria_crosshair_KR) 기반")
    ap.add_argument("image", nargs="+", help="판정할 사진 경로(여러 개 가능) 또는 폴더")
    ap.add_argument("--pose", choices=["auto", "supine", "lateral"], default="auto",
                    help="auto=자동 판별(기본) / supine=앙와위 / lateral=측위")
    ap.add_argument("--calib-from", default=None,
                    help="앙와위·침대 평평 사진 경로. 여기서 W0/L0를 재서 저장하고 사용합니다")
    ap.add_argument("--w0", type=float, default=None, help="앙와위 기준 어깨너비(픽셀) 직접 지정")
    ap.add_argument("--h0", type=float, default=None, help="앙와위 기준 골반너비(픽셀) 직접 지정")
    ap.add_argument("--l0", type=float, default=None, help="앙와위 기준 몸통길이(픽셀) 직접 지정")
    ap.add_argument("--no-calib", action="store_true", help="저장된 기준값을 무시하고 실행")
    ap.add_argument("--out", default=None, help="결과 이미지 저장 경로(사진 1장일 때만)")
    ap.add_argument("--out-dir", default=None,
                    help=f"결과 이미지를 모아 둘 폴더 (기본: {DEFAULT_OUT_DIR})")
    ap.add_argument("--beside", action="store_true",
                    help="결과를 원본 사진 옆에 저장 (원본명_result.jpg)")
    ap.add_argument("--open", dest="open_result", action="store_true",
                    help="저장한 결과 이미지를 바로 열기 (Windows)")
    ap.add_argument("--csv", default=None, help="판정 요약을 CSV로 저장(검증 재현용)")
    ap.add_argument("--api", choices=["auto", "legacy", "tasks"], default="auto",
                    help="auto=설치된 mediapipe에 맞춰 자동(기본) / legacy=0.10.x / tasks=1.0+")
    ap.add_argument("--min-avg-vis", type=float, default=MIN_AVG_VIS,
                    help=f"핵심관절 평균 신뢰도가 이 값 미만이면 판정 보류 (기본 {MIN_AVG_VIS})")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    calib = None if args.no_calib else load_calib()
    if args.calib_from:
        img = imread_u(args.calib_from)
        if img is None:
            sys.exit(f"[오류] 기준 사진을 열 수 없습니다: {args.calib_from}")
        found = extract_landmarks(img, api=args.api)
        if found is None:
            sys.exit("[오류] 기준 사진에서 사람을 찾지 못했습니다.")
        calib = measure_calibration(found[0])
        save_calib(calib)
        print(f"앙와위 기준값 저장: W0={calib['W0']:.1f}px  H0={calib['H0']:.1f}px  "
              f"L0={calib['L0']:.1f}px")
    if args.w0 or args.h0 or args.l0:
        calib = dict(calib or {})
        for key, val in (("W0", args.w0), ("H0", args.h0), ("L0", args.l0)):
            if val:
                calib[key] = val

    files = _collect(args.image)
    if not files:
        sys.exit("[오류] 판정할 사진을 찾지 못했습니다.")
    batch = len(files) > 1
    out_dir = None if args.beside else (args.out_dir or DEFAULT_OUT_DIR)
    pose_type = {"auto": "auto", "supine": "앙와위", "lateral": "측위"}[args.pose]

    print(f"mediapipe {mp.__version__} · 사진 {len(files)}장 · 자세 {args.pose}")
    print("앙와위 기준값: " + (f"W0={calib.get('W0', 0):.1f} H0={calib.get('H0', 0):.1f} "
                          f"L0={calib.get('L0', 0):.1f}"
                          if calib else "없음 (⑥ 상체 기울기·② 측위 각도는 측정 불가)"))
    print(f"결과 저장 위치: {'원본 사진 옆' if args.beside else out_dir}")

    results, used = [], set()
    for i, path in enumerate(files, 1):
        if batch:
            print(f"\n[{i}/{len(files)}] {os.path.basename(path)}")
        if args.out and not batch:
            out = args.out
        elif args.beside:
            out = os.path.splitext(path)[0] + "_result.jpg"
        else:
            out = os.path.join(out_dir, auto_name(path))
            n = 2
            while out in used:
                out = os.path.join(out_dir, auto_name(path)[:-len("_result.jpg")] + f"_{n}_result.jpg")
                n += 1
            used.add(out)
        results.append(check_image(path, out=out, calib=calib, pose_type=pose_type,
                                   min_avg_vis=args.min_avg_vis, api=args.api,
                                   open_result=args.open_result))

    if batch:
        from collections import Counter
        c = Counter(r["tag"] for r in results)
        print("\n=== 합계 ===")
        for tag in ["정상", "주의", "불량", "보류", "오류"]:
            print(f"  {tag} {c.get(tag, 0)}장")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f)
            w.writerow(["파일", "자세", "판정", "요약", "핵심관절_평균신뢰도", "결과이미지"])
            for r in results:
                w.writerow([r["path"], r["pose_type"], r["tag"], r["verdict"],
                            f"{r['avg_vis']:.3f}" if r["avg_vis"] is not None else "",
                            r["out"] or ""])
        print(f"CSV 저장: {args.csv}")

    order = ["error", "red", "hold", "yellow", "green"]
    codes = {r["code"] for r in results}
    for k in order:
        if EXIT_CODE[k] in codes:
            sys.exit(EXIT_CODE[k])


if __name__ == "__main__":
    main()