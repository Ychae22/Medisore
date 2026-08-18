import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS # ⭐️ 1. CORS 라이브러리 추가
import numpy as np
import cv2
from PIL import Image

# 커스텀 모듈 임포트
from infer import predict
from pose import analyze_posture

app = Flask(__name__)
CORS(app) # ⭐️ 2. 모든 도메인에서 이 서버로 요청할 수 있도록 허용!

@app.route('/')
# ... (이하 기존 코드 동일) ...

@app.route('/')
def home():
    return render_template('index.html')

# --- 1. 욕창 상처 분석 API ---
@app.route('/analyze', methods=['POST'])
def analyze_wound():
    if 'file' not in request.files: return jsonify({'error': '파일이 없습니다.'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        result = predict(image)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f"파이썬 에러: {str(e)}"}), 500
    
# --- 2. 자세 분석 API ---
@app.route('/analyze_pose', methods=['POST'])
def pose_endpoint():
    if 'file' not in request.files: return jsonify({'error': '파일이 없습니다.'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    pose_type = request.form.get('pose_type', 'supine')
    w0 = request.form.get('W0')
    h0 = request.form.get('H0')
    l0 = request.form.get('L0')

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_array is None: return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400

        pose_img, result_data = analyze_posture(img_array, pose_type=pose_type, W0=w0, L0=l0, H0=h0)
        
        if pose_img is None:
            return jsonify({'error': result_data['error']}), 200
            
        result_data['pose_image'] = pose_img
        return jsonify(result_data)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f"자세 분석 에러: {str(e)}"}), 500

# --- 3. 7일간 간호기록지(PDF) 생성 및 다운로드 API ---
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.pagesizes import A4
        import tempfile
    except ImportError:
        return jsonify({'error': 'PDF 라이브러리가 설치되지 않았습니다. pip install reportlab'}), 500

    data = request.json
    patient_name = data.get('patient_name', '환자')
    gender = data.get('gender', 'M')
    age = data.get('age', '')
    caregiver_id = data.get('caregiver_id', '')
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    timeline = data.get('timeline', [])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, 'malgun.ttf')
    if not os.path.exists(font_path):
        return jsonify({'error': 'malgun.ttf 폰트를 찾을 수 없습니다.'}), 500
    pdfmetrics.registerFont(TTFont('Malgun', font_path))

    def get_image_path(filename):
        paths = [os.path.join(base_dir, 'static', filename), os.path.join(base_dir, filename)]
        for p in paths:
            if os.path.exists(p): return p
        return None

    logo_path = get_image_path('logo_3.png') or get_image_path('logo.png') # 최신 로고 반영
    roti_path = get_image_path('roti_2.png') or get_image_path('roti.png')
    
    body_map_back_path = get_image_path('body_map_back.png')
    body_map_front_path = get_image_path('body_map_front.png')

    # ⭐️ 프론트엔드(index.html)에서 직접 맞춘 퍼센트 좌표와 100% 동일하게 동기화!
    BACK_COORDS = {
        '뒤통수': (0.51, 0.10), '왼쪽 견갑골': (0.39, 0.24), '오른쪽 견갑골': (0.64, 0.24),
        '왼쪽 팔꿈치': (0.20, 0.38), '오른쪽 팔꿈치': (0.83, 0.38), '척추': (0.51, 0.41),
        '천골(엉치뼈)': (0.51, 0.48), '왼쪽 대전자': (0.34, 0.50), '오른쪽 대전자': (0.70, 0.50),
        '왼쪽 좌골결절': (0.42, 0.54), '오른쪽 좌골결절': (0.62, 0.54), '왼쪽 발목(후면)': (0.42, 0.82),
        '오른쪽 발목(후면)': (0.61, 0.82), '왼쪽 발뒤꿈치': (0.43, 0.93), '오른쪽 발뒤꿈치': (0.60, 0.93)
    }
    
    FRONT_COORDS = {
        '왼쪽 귀': (0.36, 0.13), '오른쪽 귀': (0.58, 0.13), '왼쪽 어깨': (0.18, 0.22),
        '오른쪽 어깨': (0.77, 0.22), '왼쪽 흉곽': (0.40, 0.35), '오른쪽 흉곽': (0.55, 0.35),
        '왼쪽 골반': (0.27, 0.50), '오른쪽 골반': (0.68, 0.50), '왼쪽 허벅지': (0.37, 0.60),
        '오른쪽 허벅지': (0.60, 0.60), '왼쪽 발목(전면)': (0.38, 0.82), '오른쪽 발목(전면)': (0.57, 0.82),
        '왼쪽 발가락': (0.40, 0.94), '오른쪽 발가락': (0.56, 0.94)
    }

    wounded_parts = set()
    for day_data in timeline:
        for event in day_data.get('events', []):
            if event.get('type') == 'wound':
                wounded_parts.add(event.get('part', ''))

    fd, temp_path = tempfile.mkstemp(suffix='.pdf')
    os.close(fd)
    
    c = canvas.Canvas(temp_path, pagesize=A4)
    width, height = A4

    def draw_page_template(c, page_num):
        if logo_path: c.drawImage(logo_path, 40, height - 55, width=90, height=40, mask='auto') # 로고 조금 더 길게
        c.setFont('Malgun', 18)
        c.drawCentredString(width / 2.0, height - 45, f"욕창예방 자세 변경 스케줄 ({start_date} ~ {end_date})")
        c.setFont('Malgun', 11)
        c.drawString(40, height - 85, f"병실 / 이름: _________ / {patient_name} ({'M' if gender=='M' else 'F'}/{age})")
        c.drawRightString(width - 40, height - 85, f"담당 간호사/간병인: {caregiver_id}")

        y_offset = height - 105
        if page_num == 1:
            y_offset = height - 260
            if roti_path: c.drawImage(roti_path, 40, y_offset, width=150, height=150, mask='auto')
            
            # 후면 렌더링
            if body_map_back_path: 
                c.drawImage(body_map_back_path, 210, y_offset, width=70, height=150, mask='auto')
            # 전면 렌더링
            if body_map_front_path:
                c.drawImage(body_map_front_path, 290, y_offset, width=70, height=150, mask='auto')
                
            c.setStrokeColorRGB(0.86, 0.21, 0.27)
            c.setLineWidth(1.5)
            
            # 좌표 매핑 적용
            for part in wounded_parts:
                if part in BACK_COORDS:
                    lx, ty = BACK_COORDS[part]
                    px = 210 + (lx * 70)
                    py = (y_offset + 150) - (ty * 150)
                    c.circle(px, py, 5, fill=0, stroke=1)
                elif part in FRONT_COORDS:
                    lx, ty = FRONT_COORDS[part]
                    px = 290 + (lx * 70)
                    py = (y_offset + 150) - (ty * 150)
                    c.circle(px, py, 5, fill=0, stroke=1)
            
            c.setStrokeColorRGB(0, 0, 0)
            
            c.setFont('Malgun', 9)
            c.drawString(380, y_offset + 130, "1. 대상: 고위험 환자")
            c.drawString(380, y_offset + 110, "2. 활동사항:")
            c.drawString(390, y_offset + 95, "(1) 2시간마다 체위변경")
            c.drawString(390, y_offset + 80, "(2) 피부(상처) 확인")
            c.drawString(390, y_offset + 65, "(3) 기록지 서명 완료")

        y_pos = y_offset - 20
        c.setFont('Malgun', 10)
        
        cols = [40, 90, 130, 190, 400, 480, width-40]
        titles = ["날짜", "시간", "구분", "상세 내용 (자세 번호 / 상처 부위)", "완료 여부", "서명"]
        
        c.line(40, y_pos + 15, width - 40, y_pos + 15)
        c.line(40, y_pos - 10, width - 40, y_pos - 10)
        
        for i in range(len(cols)-1):
            c.drawCentredString((cols[i] + cols[i+1])/2.0, y_pos, titles[i])
        for col_x in cols:
            c.line(col_x, y_pos + 15, col_x, y_pos - 10)
        
        return y_pos - 25, cols

    y_pos, cols = draw_page_template(c, 1)

    def get_clinical_pose_name(name):
        if "왼쪽" in name or "좌측위" in name: return "좌측위"
        if "오른쪽" in name or "우측위" in name: return "우측위"
        return "앙와위"

    last_printed_date = ""

    c.setFont('Malgun', 9)
    for day_data in timeline:
        date_str = day_data.get('date', '')
        for event in day_data.get('events', []):
            if y_pos < 50:
                c.showPage()
                y_pos, cols = draw_page_template(c, 2)
                c.setFont('Malgun', 9)
                last_printed_date = ""

            c.line(40, y_pos - 5, width - 40, y_pos - 5)
            for col_x in cols: 
                c.line(col_x, y_pos + 15, col_x, y_pos - 5)

            display_date = date_str[-5:] if date_str != last_printed_date else ""
            c.drawCentredString((cols[0] + cols[1])/2.0, y_pos + 2, display_date)
            last_printed_date = date_str

            c.drawCentredString((cols[1] + cols[2])/2.0, y_pos + 2, event.get('time', ''))
            
            if event.get('type') == 'pose':
                c.drawCentredString((cols[2] + cols[3])/2.0, y_pos + 2, "체위변경")
                clinical_name = get_clinical_pose_name(event.get('name', ''))
                c.drawString(cols[3] + 10, y_pos + 2, f"{event.get('roti', '')}번 자세 ({clinical_name})")
                c.drawCentredString((cols[4] + cols[5])/2.0, y_pos + 2, "완료")
            else:
                c.drawCentredString((cols[2] + cols[3])/2.0, y_pos + 2, "상처관리")
                c.drawString(cols[3] + 10, y_pos + 2, f"[{event.get('part', '')}] {event.get('stage', '')}단계 확인")
                c.drawCentredString((cols[4] + cols[5])/2.0, y_pos + 2, event.get('action', '드레싱 교체 완료'))
                
            c.drawCentredString((cols[5] + cols[6])/2.0, y_pos + 2, caregiver_id)
            y_pos -= 20

    c.save()
    return send_file(temp_path, as_attachment=True, download_name=f"report_{start_date}_to_{end_date}.pdf")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)