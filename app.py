import streamlit as st
from auth_block import AuthBlock
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="The Cognitive Weaver", layout="wide", page_icon="💎")

# 2. KHỞI TẠO CÁC KHỐI
auth = AuthBlock()
ai = AI_Core()
voice = Voice_Engine()

# 3. MÀN HÌNH ĐĂNG NHẬP
def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("🔐 Đăng Nhập Hệ Thống")
        pwd = st.text_input("Mật khẩu truy cập:", type="password")
        if st.button("Đăng Nhập", use_container_width=True):
            if auth.login(pwd):
                st.rerun()
            else:
                st.error("Mật khẩu không đúng!")

# 4. GIAO DIỆN CHÍNH
def main_app():
    # Sidebar thông tin
    with st.sidebar:
        st.success(f"👤 User: {st.session_state.current_user}")
        if st.session_state.is_vip:
            st.info("🌟 Trạng thái: VIP (Unlimited)")
        else:
            used, limit, _ = auth.check_quota_status()
            st.progress(min(1.0, used/limit))
            st.caption(f"Quota: {used}/{limit}")
            
        if st.button("Đăng xuất"):
            st.session_state.user_logged_in = False
            st.rerun()

    st.title("💎 Người Dệt Nhận Thức (AI Weaver)")
    
    # Tabs chức năng
    t1, t2, t3 = st.tabs(["📚 Phân Tích Sách", "🗣️ Tranh Biện", "🎙️ Phòng Thu"])

    # --- TAB 1: SÁCH (RAG) ---
    with t1:
        st.header("Trợ Lý Đọc Sách")
        up_file = st.file_uploader("Upload tài liệu (Txt/PDF)...")
        if up_file and st.button("Phân tích"):
            # Kiểm tra quota (Giả sử 1 lần phân tích tốn 5000 chars)
            usage, limit, allowed = auth.check_quota_status()
            if allowed:
                try:
                    text = up_file.read().decode("utf-8", errors='ignore')
                    # Gọi AI có Cache
                    with st.spinner("AI đang đọc..."):
                        res = ai.analyze_static(text, BOOK_ANALYSIS_PROMPT)
                        st.markdown(res)
                        auth.track_usage(len(text)) # Trừ tiền
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")
            else:
                st.error("Hết Quota!")

    # --- TAB 2: TRANH BIỆN (CHAT) ---
    with t2:
        st.header("Đấu Trường Tư Duy")
        c1, c2 = st.columns([3, 1])
        with c1:
            persona = st.selectbox("Chọn Đối Thủ:", list(DEBATE_PERSONAS.keys()))
        with c2:
            if st.button("Xóa Chat"): st.session_state.history = []; st.rerun()

        if "history" not in st.session_state: st.session_state.history = []

        # Hiển thị Chat
        for msg in st.session_state.history:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Nhập luận điểm của bạn..."):
            # Check Quota
            _, _, allowed = auth.check_quota_status()
            if not allowed:
                st.error("Hết Quota ngày hôm nay!")
            else:
                st.chat_message("user").write(prompt)
                st.session_state.history.append({"role": "user", "content": prompt})
                
                with st.chat_message("assistant"):
                    with st.spinner(f"{persona} đang suy nghĩ..."):
                        # Ghép lịch sử để AI nhớ
                        context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history[-5:]])
                        full_prompt = f"LỊCH SỬ CHAT:\n{context}\n\nUSER MỚI NÓI: {prompt}"
                        
                        # Gọi AI (Dùng Flash cho nhanh)
                        reply = ai.generate(full_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[persona])
                        
                        st.write(reply)
                        st.session_state.history.append({"role": "assistant", "content": reply})
                        auth.track_usage(len(prompt) + len(reply))

    # --- TAB 3: VOICE (TTS) ---
    with t3:
        st.header("Phòng Thu AI")
        txt = st.text_area("Nhập văn bản cần đọc:")
        if st.button("Đọc Ngay"):
            with st.spinner("Đang tạo âm thanh..."):
                audio_file = voice.speak(txt, lang="vi")
                if audio_file:
                    st.audio(audio_file)
                    st.success("Xong!")

# --- ENTRY POINT ---
if __name__ == "__main__":
    if st.session_state.get('user_logged_in', False):
        main_app()
    else:
        login_screen()
