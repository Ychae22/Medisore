# -*- coding: utf-8 -*-
"""
웹(dpp.py)과 판정 모듈(pose_check.py) 사이를 잇는 얇은 층.

dpp.py 쪽 호출 형태와 응답 키는 그대로 유지합니다. 바뀐 점만 정리하면:
  · 자세(앙와위/측위)는 프론트가 보낸 pose_type 대신 사진에서 자동 판별합니다.
  · 첫 촬영이 앙와위면 기준값 W0/L0를 재서 calib_data로 돌려줍니다.
    프론트는 이 값을 보관했다가 다음 촬영 때 그대로 다시 보내주면 됩니다.
  · 결과 이미지는 문제가 잡힌 부위만 노랑·빨강으로 칠합니다.
"""
import base64
from io import BytesIO

import cv2
from PIL import Image

from pose_check import (
    extract_landmarks, judge, draw_overlay, draw_pressure, draw_guide_rail,
    measure_calibration, MIN_AVG_VIS, 표시설정,
)


def _to_b64(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_posture(img_array, pose_type="supine", W0=None, L0=None, H0=None):
    """사진 1장을 판정해서 (오버레이 이미지 base64, 결과 딕셔너리)를 돌려줍니다.

    H0(앙와위 골반너비)는 나중에 추가된 값이라 맨 뒤에 뒀습니다. 안 넘겨도 동작하지만,
    없으면 자세 판별이 근사로 떨어지고 상체·하체 비틀림은 계산되지 않습니다.
    """
    try:
        # want_mask=True — 사람 실루엣은 히트맵 표시에만 씁니다(판정에는 안 씀).
        found = extract_landmarks(img_array, want_mask=True)
        if found is None:
            return None, {'error': '사람을 찾지 못했습니다. 몸 전체가 보이게 다시 촬영해 주세요.'}

        pts2d, pts3d, low_vis, avg_vis, vis, body_mask = found

        # 프론트에서 넘어온 앙와위 기준값(있으면 사용)
        calib = {}
        for key, val in (("W0", W0), ("H0", H0), ("L0", L0)):
            try:
                if val is not None and float(val) > 0:
                    calib[key] = float(val)
            except (TypeError, ValueError):
                pass
        calib = calib or None

        # 자세는 자동 판별 (프론트의 pose_type은 참고용으로만 응답에 실어 보냄)
        # pts3d 는 좌/우 판별이 애매할 때 보조 신호(깊이)로만 씁니다 — pose_check.py 자세판별 주석 참고
        result = judge(pts2d, vis, calib=calib, pose_type="auto", pts3d=pts3d,
                       shape=img_array.shape)

        # 측위면 바닥에 닿은 쪽까지 붙여서 화면 표시용 라벨을 만듭니다.
        #   down_side "l" = 왼쪽 어깨가 바닥 = 좌측위 / "r" = 우측위
        #   None 이면 좌우 판별 보류 — 라벨은 "측위" 로 둡니다.
        down = result.get("down_side")
        pose_label = result["pose_type"]
        if pose_label == "측위" and down:
            pose_label = "좌측위" if down == "l" else "우측위"

        # 판정 보류 두 가지 — ① 이불 가림 ② 상체와 하체가 다른 방향으로 비틀림
        #   비틀림은 앙와위 기준표도 측위 기준표도 맞지 않는 상태라 등급을 내지 않습니다.
        covered = avg_vis < MIN_AVG_VIS
        twisted = bool(result["twist_hold"])
        hold = covered or twisted
        hold_reason = "가림" if covered else ("비틀림" if twisted else None)
        hold_message = ("이불을 걷거나 몸이 보이게 다시 촬영해 주세요" if covered else
                        "상체와 하체가 같은 쪽을 보도록 다시 눕혀 주세요" if twisted else None)

        # 앙와위인데 기준값이 아직 없으면 이번 사진으로 재서 돌려줍니다(첫 촬영 캘리브레이션)
        calib_out = calib
        if result["pose_type"] == "앙와위" and not calib and not hold:
            calib_out = measure_calibration(pts2d)

        # 예상 압력 히트맵 — 측정이 아니라 예측입니다(pose_check.py 압력예상 주석 참고).
        # 화면에 '예상'이라고 반드시 밝혀 주세요.
        #   ※ draw_overlay 가 원본 배열에 직접 그리므로 반드시 먼저 뽑습니다.
        heat_cv, heat_key = draw_pressure(img_array.copy(), pts2d, vis, result["pose_type"],
                                          roll=result["roll"]["하체각"],
                                          down=result.get("down_side"), mask=body_mask)
        
        # 보류(hold)면 안내를 못 믿으므로 레일을 안 붙입니다.
        # 프론트엔드 CSS로 직접 처리하기 위해 서버 사이드 렌더링을 끕니다.
        # if not hold:
        #     heat_cv = draw_guide_rail(heat_cv, result["guides"])
        pressure_img_base64 = _to_b64(heat_cv)

        # 사진 위에 뼈대 그리기 — 문제 부위만 색이 바뀝니다
        result_img_cv = draw_overlay(img_array, pts2d, result["status"],
                                     hold=hold, parts=result["parts"])

        pose_img_base64 = _to_b64(result_img_cv)

        result_data = {
            'status': result["status"],        # green / yellow / red
            'verdict': result["verdict"],
            'lines': result["lines"],
            # ↓ 간병인 화면은 이 두 개만 쓰면 됩니다. 각도가 안 들어갑니다.
            'guides': result["guides"],            # [{item, status, bad, do, why, parts}]
            'good_points': result["good_points"],  # 정상 항목 칭찬 문장
            'guide_max': 3,                        # 한 번에 보여줄 개수 (나머지는 접기)
            'hold': hold,
            'hold_reason': hold_reason,        # "가림" / "비틀림" / None
            'hold_message': hold_message,      # 보류일 때 화면에 띄울 안내 문구
            # 상체(어깨)·하체(골반) 회전각과 그 차이. 기준값이 없으면 값이 None 입니다.
            'roll': result["roll"],            # {"기준", "어깨각", "골반각", "비틀림"}
            # 보조기구 안내 — ⑦ 무릎 신전 / ⑧ 족하수 중 하나라도 정상이 아니면 needed=True.
            # 자세 등급(status)과는 별개로 화면에 따로 띄우면 됩니다.
            'support': result["support"],
            'pose_type': result["pose_type"],  # 자동 판별한 자세: "앙와위" / "측위"
            # ↓ 임상 표기: "앙와위" / "좌측위" / "우측위" / "측위"(좌우 미상)
            'pose_label': pose_label,
            # ↓ 간병인 화면에 띄울 값: "바로 누움" / "왼쪽으로 누움" / "오른쪽으로 누움"
            'pose_label_user': 표시설정["자세_사용자표기"].get(pose_label, pose_label),
            'pose_side': down,                 # "l"(좌) / "r"(우) / None(판별 보류)
            'pose_type_sent': pose_type,       # 프론트가 보낸 값 (참고용)
            'warning': result["warning"],      # 촬영 방향이 어긋나면 안내 문구, 아니면 None
            'calib_data': calib_out,           # {"W0":..., "H0":..., "L0":...} — 프론트가 보관했다가 재전송
            # 예상 압력 히트맵 (base64). 측정값이 아니라 자세로부터의 예측입니다.
            'pressure_image': pressure_img_base64,
            'pressure_zone': heat_key,         # "앙와위" / "측위_얕음" / "측위_보통" / "측위_깊음"
            'pressure_note': "압력 센서로 잰 값이 아니라 자세로부터 예상한 분포입니다",
        }
        return pose_img_base64, result_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, {'error': f'자세 분석 중 오류 발생: {str(e)}'}