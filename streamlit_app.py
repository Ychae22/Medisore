import streamlit as st
import requests
import datetime
import pandas as pd
from io import BytesIO

# --- ⚙️ 페이지 설정 ---
st.set_page_config(page_title="medisore - 욕창 관리", page_icon="🛏️", layout="centered")

# --- 💾 세션 상태 (전역 변수/LocalStorage 대체) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'patient_registered' not in st.session_state:
    st.session_state.patient_registered = False
if 'timeline' not in st.session_state:
    st.session_state.timeline = [] # 활동 기록 저장
if 'last_pose' not in st.session_state:
    st.session_state.last_pose = "바로 누움"

# 🌐 API 서버 주소
API_BASE_URL = "https://medisore-api.onrender.com"

# --- 1️⃣ 로그인 화면 ---
if not st.session_state.logged_in:
    st.image("https://raw.githubusercontent.com/Ychae22/Medisore/main/static/logo_2_3.png", width=120)
    st.title("사용자 로그인")
    st.write("스마트한 욕창 관리의 시작")
    
    with st.form("login_form"):
        c_id = st.text_input("아이디 (사번)")
        c_pw = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인", use_container_width=True)
        
        if submitted:
            if c_id and c_pw:
                st.session_state.caregiver_id = c_id
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("아이디와 비밀번호를 모두 입력해주세요.")
    st.stop()

# --- 2️⃣ 환자 등록 화면 ---
if not st.session_state.patient_registered:
    st.title("환자 등록")
    st.write("맞춤형 체위 추천을 위해 정보를 입력해주세요.")
    
    with st.form("patient_form"):
        p_name = st.text_input("환자 이름")
        p_age = st.number_input("환자 나이 (만)", min_value=0, max_value=120, step=1)
        p_gender = st.selectbox("성별", ["남성", "여성"])
        submitted = st.form_submit_button("시작하기", use_container_width=True)
        
        if submitted:
            if p_name and p_age:
                st.session_state.patient_name = p_name
                st.session_state.patient_age = p_age
                st.session_state.patient_gender = p_gender
                st.session_state.patient_registered = True
                st.rerun()
            else:
                st.error("이름과 나이를 입력해주세요.")
    st.stop()

# ==========================================
# 📱 메인 앱 화면 (로그인 & 환자 등록 완료 후)
# ==========================================

# 상단 헤더
st.markdown(f"### 안녕하세요, **{st.session_state.caregiver_id}** 간병인님 👋")
st.caption(f"환자: {st.session_state.patient_name} ({st.session_state.patient_age}세 / {st.session_state.patient_gender})")

# 하단 네비게이션을 대체하는 Streamlit Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 일정", "🧍 체위 추천", "🔍 단계 판별", "🩹 가이드", "📄 기록"])

# --- TAB 1: 홈 (일정) ---
with tab1:
    st.subheader("오늘의 추천 체위 변경")
    
    pose_cycle = ["바로 누움", "왼쪽으로 누움", "오른쪽으로 누움"]
    curr_idx = pose_cycle.index(st.session_state.last_pose) if st.session_state.last_pose in pose_cycle else 0
    next_pose = pose_cycle[(curr_idx + 1) % 3]
    
    st.info(f"**현재 자세:** {st.session_state.last_pose} ➡️ **다음 추천 자세:** {next_pose}")
    
    st.write("---")
    st.write("🕒 체위 변경 간격 설정")
    interval = st.slider("시간 간격", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    st.success(f"{interval}시간마다 알람 및 기록이 세팅되었습니다.")

# --- TAB 2: 체위 추천 (Pose) ---
with tab2:
    st.subheader("📸 자세 분석 / 코칭")
    st.info("침대를 평평하게 내리고, 팔을 몸 옆에 둔 채 환자의 전신을 촬영해주세요.")
    
    pose_img = st.file_uploader("환자 자세 사진 업로드", type=['png', 'jpg', 'jpeg'], key="pose")
    
    if pose_img is not None:
        st.image(pose_img, caption="업로드된 사진", use_column_width=True)
        if st.button("AI 정렬 분석 시작", use_container_width=True, type="primary"):
            with st.spinner("AI가 자세를 분석 중입니다..."):
                files = {'file': pose_img.getvalue()}
                data = {'pose_type': 'auto'}
                try:
                    res = requests.post(f"{API_BASE_URL}/analyze_pose", files=files, data=data)
                    result = res.json()
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success(f"분석 완료: {result.get('pose_label_user', '알 수 없음')}")
                        if 'warning' in result and result['warning']:
                            st.warning(result['warning'])
                            
                        # 결과 기록 버튼
                        if st.button("이 자세를 리포트에 기록하기", key="save_pose"):
                            st.session_state.timeline.append({
                                'time': datetime.datetime.now().strftime("%H:%M"),
                                'type': '자세 변경',
                                'detail': result.get('pose_label_user', '기록됨'),
                                'status': '완료'
                            })
                            st.session_state.last_pose = result.get('pose_label_user', next_pose)
                            st.success("기록되었습니다!")
                except Exception as e:
                    st.error(f"서버 통신 에러: {e}")

# --- TAB 3: 상처 분석 (Wound) ---
with tab3:
    st.subheader("🔍 욕창 단계 판정 보조")
    
    # 인체도 클릭 대신 Streamlit에서는 Selectbox가 가장 깔끔합니다.
    body_parts = ["선택 안함", "뒤통수", "견갑골", "팔꿈치", "천골(엉치뼈)", "대전자", "발목", "발뒤꿈치", "기타"]
    selected_part = st.selectbox("욕창 발생 의심 부위를 선택해주세요.", body_parts)
    
    if selected_part != "선택 안함":
        st.write(f"**선택된 부위:** {selected_part}")
        wound_img = st.file_uploader("상처 사진 업로드", type=['png', 'jpg', 'jpeg'], key="wound")
        
        if wound_img is not None:
            st.image(wound_img, caption="업로드된 상처", width=300)
            if st.button("AI 단계 판별하기", use_container_width=True, type="primary"):
                with st.spinner("분석 중입니다..."):
                    files = {'file': wound_img.getvalue()}
                    try:
                        res = requests.post(f"{API_BASE_URL}/analyze", files=files)
                        result = res.json()
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            st.success(f"판정 결과: {result['label']}")
                            st.info(result['message'])
                            
                            st.write("각 단계별 확률:")
                            st.json(result['probs'])
                            
                            st.warning("⚠️ 이 앱은 의료기기가 아니며, 최종 판단은 전문 의료진이 해야 합니다.")
                    except Exception as e:
                        st.error(f"서버 통신 에러: {e}")

# --- TAB 4: 처치 가이드 ---
with tab4:
    st.subheader("🩹 단계별 처치법")
    
    with st.expander("✅ 1단계 (비 창백성 홍반)"):
        st.markdown("""
        1. **압력 요인 제거**: 즉각적인 체위 변경이 필수적입니다.
        2. **예방적 드레싱 적용**:
            * 접착성 폼 드레싱 패치 (삼출물 흡수)
            * 하이드로콜로이드 밴드
        """)
        
    with st.expander("✅ 2단계 (부분층 피부 손상)"):
        st.markdown("""
        1. **상처 부위 청결 유지**: 생리식염수로 세척 후 건조
        2. **적절한 드레싱 적용**:
            * 하이드로겔 드레싱 패치 (수분 유지)
            * 실버 드레싱 폼 (감염 억제)
        """)
        
    with st.expander("🚨 3~4단계 (전층 피부 손상 - 즉시 병원 방문!)", expanded=True):
        st.error("감염 위험이 매우 높습니다. 자가 치료를 멈추고 반드시 전문 의료진의 진료를 받으세요.")

# --- TAB 5: 기록 및 리포트 ---
with tab5:
    st.subheader("📄 기록 및 리포트")
    
    if len(st.session_state.timeline) == 0:
        st.info("아직 기록된 데이터가 없습니다.")
    else:
        # 타임라인을 표(Dataframe)로 보여주기
        df = pd.DataFrame(st.session_state.timeline)
        st.dataframe(df, use_container_width=True)
        
        # 껍데기 다운로드 버튼 (API 구조에 맞춰 추후 연동 가능)
        if st.button("📥 리포트 다운로드 (PDF)"):
            st.success("API 서버에 리포트 생성을 요청했습니다! (현재는 Streamlit에서 표기로 대체됩니다.)")