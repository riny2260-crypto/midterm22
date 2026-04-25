import streamlit as st
from supabase import create_client, Client
import pandas as pd
import easyocr
from PIL import Image
import numpy as np
import re
import fitz  # PyMuPDF

# 1. Supabase 설정
# Streamlit Cloud의 Settings > Secrets에 [supabase] 섹션이 정확히 있어야 합니다.
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase: Client = create_client(url, key)


# 2. OCR 모델 로드 (캐시 사용)
# gpu=False는 무료 서버 환경에서 필수 설정입니다.
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['ko', 'en'], gpu=False)


# 3. 데이터베이스 로드 (NULL 처리 포함)
def get_db_data():
    try:
        response = supabase.table("teacher_list").select("name, is_submitted").execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            # Supabase의 NULL 값을 미제출(False)로 변환
            df['is_submitted'] = df['is_submitted'].fillna(False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


# 페이지 설정
st.set_page_config(page_title="이수증 제출 시스템", layout="wide", page_icon="🛡️")
st.title("🛡️ 이수증 자동 제출 시스템")

teacher_df, error_msg = get_db_data()

if error_msg:
    st.error(f"⚠️ 데이터베이스 연결 실패: {error_msg}")

# 4. 실시간 현황 UI
if not teacher_df.empty:
    st.subheader("📊 실시간 제출 현황")
    total = len(teacher_df)
    done = len(teacher_df[teacher_df['is_submitted'] == True])
    unsubmitted = teacher_df[teacher_df['is_submitted'] == False]['name'].tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("총 인원", f"{total}명")
    c2.metric("제출 완료", f"{done}명")
    c3.metric("제출률", f"{(done / total * 100):.1f}%" if total > 0 else "0%")

    if unsubmitted:
        st.error(f"❌ 미제출자: {', '.join(unsubmitted)}")
    else:
        st.success("🎉 모든 선생님이 제출하셨습니다!")

    st.divider()

    # 5. 파일 업로드 및 분석 (지연 로딩 전략)
    st.subheader("📸 이수증 업로드")
    uploaded_files = st.file_uploader("이수증 파일을 선택하세요 (PDF, JPG, PNG)",
                                      type=['pdf', 'jpg', 'jpeg', 'png'],
                                      accept_multiple_files=True)

    if uploaded_files:
        # 파일을 올렸을 때만 AI 모델을 불러와 메모리 과부하 방지
        with st.spinner("AI 분석 엔진을 깨우는 중입니다... (최초 1회 소요)"):
            reader = load_ocr_model()

        results = []
        for uploaded_file in uploaded_files:
            try:
                # PDF/이미지 변환
                if uploaded_file.type == "application/pdf":
                    pdf_bytes = uploaded_file.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page = doc.load_page(0)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                else:
                    image = Image.open(uploaded_file)

                image_np = np.array(image)
                ocr_results = reader.readtext(image_np, detail=1)

                detected_name = "미확인"
                all_texts = [res[1] for res in ocr_results]

                # --- 인식 로직 1: 성명 레이블 위치 기반 ---
                label_box = None
                for (bbox, text, prob) in ocr_results:
                    clean_t = re.sub(r'[^가-힣]', '', text)
                    if any(k in clean_t for k in ['성명', '이름']):
                        label_box = bbox
                        break

                if label_box:
                    label_y = (label_box[0][1] + label_box[2][1]) / 2
                    label_x_r = label_box[1][0]
                    for (bbox, text, prob) in ocr_results:
                        t_clean = re.sub(r'[^가-힣]', '', text)
                        if 2 <= len(t_clean) <= 4 and t_clean not in ['성명', '이름']:
                            t_y = (bbox[0][1] + bbox[2][1]) / 2
                            # 성명 레이블과 같은 줄(Y축 차이 50 이내)인 경우
                            if abs(label_y - t_y) < 50 and bbox[0][0] >= label_x_r - 20:
                                # 명단과 대조 (오타 보정: 3자 중 2자 일치 시 성공)
                                for real_name in teacher_df['name'].values:
                                    if sum(1 for a, b in zip(t_clean[:3], real_name[:3]) if a == b) >= 2:
                                        detected_name = real_name
                                        break
                        if detected_name != "미확인": break

                # --- 인식 로직 2: 전체 텍스트 훑기 (위치 실패 시) ---
                if detected_name == "미확인":
                    for real_name in teacher_df['name'].values:
                        for raw_text in all_texts:
                            clean_raw = re.sub(r'[^가-힣]', '', raw_text)
                            if real_name in clean_raw or (len(clean_raw) >= 2 and sum(
                                    1 for a, b in zip(clean_raw[:3], real_name[:3]) if a == b) >= 2):
                                detected_name = real_name
                                break
                        if detected_name != "미확인": break

                # DB 업데이트
                if detected_name != "미확인":
                    supabase.table("teacher_list").update({"is_submitted": True}).eq("name", detected_name).execute()
                    status = f"✅ {detected_name} 선생님 확인"
                else:
                    status = "❓ 인식 실패"
                    with st.expander(f"🔍 '{uploaded_file.name}' 분석 로그"):
                        st.write(f"추출된 텍스트 전체: {', '.join(all_texts)}")

                results.append({"파일명": uploaded_file.name, "인식 성명": detected_name, "결과": status})

            except Exception as e:
                st.error(f"파일 처리 오류: {e}")

        if results:
            st.table(pd.DataFrame(results))
            if st.button("제출 현황 새로고침"):
                st.rerun()