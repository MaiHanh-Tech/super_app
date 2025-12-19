import streamlit as st
import time
from ai_core import AI_Engine           # Block AI
from prompts import DEBATE_PERSONAS     # Block Nội dung (Tư duy của Anh)
from password_manager import PasswordManager # Block Bảo mật

# ... (Phần cấu hình trang và hàm phụ trợ giữ nguyên) ...

# --- 5. GIAO DIỆN CHÍNH ---
def show_main_app():
    # Khởi tạo Engine
    ai = AI_Engine()

    st.title("🕸️ The Cognitive Weaver")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Sách", "✍️ Dịch", "🗣️ Tranh Biện", "🎙️ Studio", "⏳ Nhật Ký"])

    # ... (Tab 1, Tab 2 giữ nguyên) ...

    # === TAB 3: ĐẤU TRƯỜNG TƯ DUY & THÚC THÚC ===
    with tab3:
        st.header("Phòng Tranh Biện & Cố Vấn Ảo")
        
        # Chọn Nhân cách (Lấy từ file prompts.py)
        c1, c2 = st.columns([3, 1])
        with c1:
            persona_name = st.selectbox("Chọn Người Đối Thoại:", list(DEBATE_PERSONAS.keys()))
        with c2:
            if st.button("🗑️ Xóa Chat"):
                st.session_state.chat_history = []
                st.rerun()

        # Lấy System Prompt tương ứng
        selected_system_prompt = DEBATE_PERSONAS[persona_name]

        # Hiển thị lịch sử chat
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if user_input := st.chat_input("Nhập vấn đề cần phân tích/tranh biện..."):
            # 1. Hiện câu hỏi user
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # 2. Gọi AI với System Prompt đặc biệt
            with st.chat_message("assistant"):
                with st.spinner(f"{persona_name} đang suy ngẫm..."):
                    # Ghép lịch sử chat để AI nhớ ngữ cảnh
                    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
                    
                    full_prompt = f"""
                    LỊCH SỬ TRÒ CHUYỆN:
                    {history_context}
                    
                    CÂU HỎI MỚI NHẤT: {user_input}
                    
                    HÃY TRẢ LỜI VỚI TƯ CÁCH LÀ: {persona_name}
                    """
                    
                    # Gọi AI Core (Dùng Pro model cho Thúc thúc để sâu sắc hơn)
                    use_pro = "Thúc Thúc" in persona_name
                    response = ai.generate_content(full_prompt, system_instruction=selected_system_prompt, use_pro=use_pro)
                    
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ... (Các Tab khác giữ nguyên) ...

# ... (Phần Main giữ nguyên) ...
