import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.express as px
import time
from datetime import datetime
import json
import re

# ✅ THAY GOOGLE SHEETS BẰNG SUPABASE
from supabase import create_client, Client

# --- IMPORT CÁC META-BLOCKS ---
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# ==========================================
# 🌍 BỘ TỪ ĐIỂN ĐA NGÔN NGỮ
# ==========================================
TRANS = {
    "vi": {
        "lang_select": "Ngôn ngữ / Language / 语言",
        "tab1": "📚 Phân Tích Sách",
        "tab2": "✍️ Dịch Giả",
        "tab3": "🗣️ Tranh Biện",
        "tab4": "🎙️ Phòng Thu AI",
        "tab5": "⏳ Nhật Ký",
        "t1_header": "Trợ lý Nghiên cứu & Knowledge Graph",
        "t1_up_excel": "1. Kết nối Kho Sách (Excel)",
        "t1_up_doc": "2. Tài liệu mới (PDF/Docx)",
        "t1_btn": "🚀 PHÂN TÍCH NGAY",
        "t1_analyzing": "Đang phân tích {name}...",
        "t1_connect_ok": "✅ Đã kết nối {n} cuốn sách.",
        "t1_graph_title": "🪐 Vũ Trụ Sách",
        "t2_header": "Dịch Thuật Đa Chiều",
        "t2_input": "Nhập văn bản cần dịch:",
        "t2_target": "Dịch sang:",
        "t2_style": "Phong cách:",
        "t2_btn": "✍️ Dịch Ngay",
        "t3_header": "Đấu Trường Tư Duy",
        "t3_persona_label": "Chọn Đối Thủ:",
        "t3_input": "Nhập chủ đề tranh luận...",
        "t3_clear": "🗑️ Xóa Chat",
        "t4_header": "🎙️ Phòng Thu AI Đa Ngôn Ngữ",
        "t4_voice": "Chọn Giọng:",
        "t4_speed": "Tốc độ:",
        "t4_btn": "🔊 TẠO AUDIO",
        "t5_header": "Nhật Ký & Lịch Sử",
        "t5_refresh": "🔄 Tải lại Lịch sử",
        "t5_empty": "Chưa có dữ liệu lịch sử.",
    },
    "en": {
        "lang_select": "Language",
        "tab1": "📚 Book Analysis",
        "tab2": "✍️ Translator",
        "tab3": "🗣️ Debater",
        "tab4": "🎙️ AI Studio",
        "tab5": "⏳ History",
        "t1_header": "Research Assistant & Knowledge Graph",
        "t1_up_excel": "1. Connect Book Database (Excel)",
        "t1_up_doc": "2. New Documents (PDF/Docx)",
        "t1_btn": "🚀 ANALYZE NOW",
        "t1_analyzing": "Analyzing {name}...",
        "t1_connect_ok": "✅ Connected {n} books.",
        "t1_graph_title": "🪐 Book Universe",
        "t2_header": "Multidimensional Translator",
        "t2_input": "Enter text to translate:",
        "t2_target": "Translate to:",
        "t2_style": "Style:",
        "t2_btn": "✍️ Translate",
        "t3_header": "Thinking Arena",
        "t3_persona_label": "Choose Opponent:",
        "t3_input": "Enter debate topic...",
        "t3_clear": "🗑️ Clear Chat",
        "t4_header": "🎙️ Multilingual AI Studio",
        "t4_voice": "Select Voice:",
        "t4_speed": "Speed:",
        "t4_btn": "🔊 GENERATE AUDIO",
        "t5_header": "Logs & History",
        "t5_refresh": "🔄 Refresh History",
        "t5_empty": "No history data found.",
    },
    "zh": {
        "lang_select": "语言",
        "tab1": "📚 书籍分析",
        "tab2": "✍️ 翻译专家",
        "tab3": "🗣️ 辩论场",
        "tab4": "🎙️ AI 录音室",
        "tab5": "⏳ 历史记录",
        "t1_header": "研究助手 & 知识图谱",
        "t1_up_excel": "1. 连接书库 (Excel)",
        "t1_up_doc": "2. 上传新文档 (PDF/Docx)",
        "t1_btn": "🚀 立即分析",
        "t1_analyzing": "正在分析 {name}...",
        "t1_connect_ok": "✅ 已连接 {n} 本书。",
        "t1_graph_title": "🪐 书籍宇宙",
        "t2_header": "多维翻译",
        "t2_input": "输入文本:",
        "t2_target": "翻译成:",
        "t2_style": "风格:",
        "t2_btn": "✍️ 翻译",
        "t3_header": "思维竞技场",
        "t3_persona_label": "选择对手:",
        "t3_input": "输入辩论主题...",
        "t3_clear": "🗑️ 清除聊天",
        "t4_header": "🎙️ AI 多语言录音室",
        "t4_voice": "选择声音:",
        "t4_speed": "语速:",
        "t4_btn": "🔊 生成音频",
        "t5_header": "日志 & 历史",
        "t5_refresh": "🔄 刷新历史",
        "t5_empty": "暂无历史数据。",
    }
}

# Hàm lấy text theo ngôn ngữ
def T(key):
    lang = st.session_state.get('weaver_lang', 'vi')
    return TRANS.get(lang, TRANS['vi']).get(key, key)

# ==========================================
# 🔄 THAY ĐỔI CHÍNH: KẾT NỐI SUPABASE
# ==========================================

@st.cache_resource
def get_supabase_client() -> Client:
    """Kết nối Supabase (chỉ chạy 1 lần)"""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Không kết nối được Supabase: {e}")
        return None

def luu_lich_su(loai: str, tieu_de: str, noi_dung: str):
    """
    Lưu lịch sử vào Supabase
    Mapping cột:
    - Time → created_at (tự động)
    - Type → type
    - Title → title
    - Content → content
    - User → user_name
    - SentimentScore → sentiment_score
    - SentimentLabel → sentiment_label
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return
        
        user = st.session_state.get("current_user", "Unknown")
        
        # ✅ Dữ liệu đúng cấu trúc bảng Supabase
        data = {
            "type": loai,
            "title": tieu_de,
            "content": noi_dung,
            "user_name": user,
            "sentiment_score": 0.0,  # Placeholder (có thể tích hợp sentiment analysis)
            "sentiment_label": "Neutral"
        }
        
        response = supabase.table("History_Logs").insert(data).execute()
        
        # Kiểm tra lỗi
        if hasattr(response, 'error') and response.error:
            st.warning(f"⚠️ Lỗi lưu lịch sử: {response.error}")
            
    except Exception as e:
        st.warning(f"⚠️ Không lưu được lịch sử: {e}")

def tai_lich_su():
    """
    Tải lịch sử từ Supabase
    Trả về danh sách dict với các key giống Google Sheets cũ
    để giữ nguyên logic hiển thị
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return []
        
        # ✅ Lấy dữ liệu, sắp xếp theo thời gian mới nhất
        response = supabase.table("History_Logs")\
            .select("*")\
            .order("created_at", desc=True)\
            .execute()
        
        if hasattr(response, 'data') and response.data:
            # ✅ Chuyển đổi tên cột để tương thích với code cũ
            return [
                {
                    "Time": item.get("created_at", ""),
                    "Type": item.get("type", ""),
                    "Title": item.get("title", ""),
                    "Content": item.get("content", ""),
                    "User": item.get("user_name", ""),
                    "SentimentScore": item.get("sentiment_score", 0.0),
                    "SentimentLabel": item.get("sentiment_label", "Neutral")
                }
                for item in response.data
            ]
        
        return []
        
    except Exception as e:
        st.error(f"❌ Lỗi tải lịch sử: {e}")
        return []

# --- CÁC HÀM PHỤ TRỢ (GIỮ NGUYÊN) ---
@st.cache_resource
def load_models():
    """Chỉ load khi thực sự cần, và giới hạn 1 instance"""
    try:
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device='cpu'
        )
        model.max_seq_length = 128
        return model
    except Exception as e:
        st.error(f"Không load được model: {e}")
        return None

def check_model_available():
    """Kiểm tra model có sẵn không trước khi dùng"""
    model = load_models()
    if model is None:
        st.warning("⚠️ Chức năng Knowledge Graph tạm thời không khả dụng (thiếu RAM)")
        return False
    return True

def doc_file(uploaded_file):
    if not uploaded_file: return ""
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif ext == "docx":
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in ["txt", "md", "html"]:
            return str(uploaded_file.read(), "utf-8")
    except: return ""
    return ""

# --- HÀM CHÍNH: RUN() (GIỮ NGUYÊN LOGIC, CHỈ THAY GỌI HÀM LƯU/TẢI) ---
def run():
    # 1. Khởi tạo các Block
    ai = AI_Core()
    voice = Voice_Engine()
    
    # 2. Sidebar chọn ngôn ngữ cho Module này
    with st.sidebar:
        st.markdown("---")
        lang_choice = st.selectbox(
            "🌐 " + TRANS['vi']['lang_select'],
            ["Tiếng Việt", "English", "中文"],
            index=0,
            key="weaver_lang_selector"
        )
        if lang_choice == "Tiếng Việt": st.session_state.weaver_lang = 'vi'
        elif lang_choice == "English": st.session_state.weaver_lang = 'en'
        elif lang_choice == "中文": st.session_state.weaver_lang = 'zh'
    
    st.header(f"🧠 The Cognitive Weaver")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")
    ])

    # === TAB 1: RAG & GRAPH ===
    with tab1:
        st.subheader(T("t1_header"))
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: file_excel = st.file_uploader(T("t1_up_excel"), type="xlsx", key="w_t1_ex")
        with c2: uploaded_files = st.file_uploader(T("t1_up_doc"), type=["pdf", "docx", "txt"], accept_multiple_files=True, key="w_t1_doc")
        with c3: 
            st.write("")
            st.write("")
            btn_run = st.button(T("t1_btn"), type="primary", use_container_width=True)

        if btn_run and uploaded_files:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            vec = load_models()
            db, df = None, None
            has_db = False
            
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                    db = vec.encode([f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in df.iterrows()])
                    has_db = True
                    st.success(T("t1_connect_ok").format(n=len(df)))
                except: st.error("Lỗi đọc Excel.")

            for file_idx, f in enumerate(uploaded_files):
                status_text.text(f"Đang xử lý file {file_idx+1}/{total_files}: {f.name}")
                progress_bar.progress((file_idx) / total_files)
                
                text = doc_file(f)
                link = ""
                if has_db:
                    q = vec.encode([text[:2000]])
                    sc = cosine_similarity(q, db)[0]
                    idx_sim = np.argsort(sc)[::-1][:3]
                    for i in idx_sim:
                        if sc[i] > 0.35: link += f"- {df.iloc[i]['Tên sách']} ({sc[i]*100:.0f}%)\n"

                with st.spinner(T("t1_analyzing").format(name=f.name)):
                    prompt = f"Phân tích tài liệu '{f.name}'. Liên quan: {link}\nNội dung: {text[:30000]}"
                    res = ai.analyze_static(prompt, BOOK_ANALYSIS_PROMPT)
                    
                    st.markdown(f"### 📄 {f.name}")
                    st.markdown(res)
                    st.markdown("---")
                    
                    # ✅ GỌI HÀM MỚI
                    luu_lich_su("Phân Tích Sách", f.name, res[:200])
                
                progress_bar.progress((file_idx+1) / total_files)
            
            status_text.text("✅ Hoàn thành!")

        # VẼ GRAPH (AGRAPH)
        if file_excel:
            try:
                with st.expander(T("t1_graph_title"), expanded=False):
                    vec = load_models()
                    if "book_embs" not in st.session_state:
                         st.session_state.book_embs = vec.encode(df["Tên sách"].tolist())
                    
                    embs = st.session_state.book_embs
                    sim = cosine_similarity(embs)
                    nodes, edges = [], []
                    
                    max_nodes = st.slider("Max Nodes:", 5, len(df), min(50, len(df)))
                    threshold = st.slider("Threshold:", 0.0, 1.0, 0.45)

                    for i in range(max_nodes):
                        nodes.append(Node(id=str(i), label=df.iloc[i]["Tên sách"], size=20, color="#FFD166"))
                        for j in range(i+1, max_nodes):
                            if sim[i,j]>threshold: edges.append(Edge(source=str(i), target=str(j), color="#118AB2"))
                    
                    config = Config(width=900, height=600, directed=False, physics=True, collapsible=False)
                    agraph(nodes, edges, config)
            except: pass

    # === TAB 2: DỊCH GIẢ ===
    with tab2:
        st.subheader(T("t2_header"))
        txt = st.text_area(T("t2_input"), height=150, key="w_t2_inp")
        c_l, c_s, c_b = st.columns([1,1,1])
        with c_l: target_lang = st.selectbox(T("t2_target"), ["Tiếng Việt", "English", "Chinese", "French", "Japanese"], key="w_t2_lang")
        with c_s: style = st.selectbox(T("t2_style"), ["Default", "Academic", "Literary", "Business"], key="w_t2_style")
        
        if st.button(T("t2_btn"), key="w_t2_btn") and txt:
            with st.spinner("AI Translating..."):
                p = f"Translate to {target_lang}. Style: {style}. Text: {txt}"
                res = ai.generate(p, model_type="pro")
                st.markdown(res)
                
                # ✅ GỌI HÀM MỚI
                luu_lich_su("Dịch Thuật", f"{target_lang}", txt[:50])

    # === TAB 3: ĐẤU TRƯỜNG TƯ DUY ===
    with tab3:
        st.subheader(T("t3_header"))
        mode = st.radio("Mode:", ["👤 Solo", "⚔️ Multi-Agent"], horizontal=True, key="w_t3_mode")
        
        if "weaver_chat" not in st.session_state: 
            st.session_state.weaver_chat = []

        if mode == "👤 Solo":
            c1, c2 = st.columns([3, 1])
            
            with c1: 
                persona = st.selectbox(
                    T("t3_persona_label"), 
                    list(DEBATE_PERSONAS.keys()), 
                    key="w_t3_solo_p"
                )
            
            with c2: 
                if st.button(T("t3_clear"), key="w_t3_clr"): 
                    st.session_state.weaver_chat = []
                    st.rerun()

            for msg in st.session_state.weaver_chat:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input(T("t3_input")):
                st.chat_message("user").write(prompt)
                st.session_state.weaver_chat.append({
                    "role": "user", 
                    "content": prompt
                })
                
                recent_history = st.session_state.weaver_chat[-10:]
                context_text = "\n".join([
                    f"{m['role'].upper()}: {m['content']}" 
                    for m in recent_history
                ])
                
                full_prompt = f"""
                LỊCH SỬ HỘI THOẠI:
                {context_text}

                NHIỆM VỤ: Dựa vào lịch sử trên, hãy trả lời câu hỏi mới nhất của USER.
                """
                
                with st.chat_message("assistant"):
                    sys_instruction = DEBATE_PERSONAS[persona]
                    
                    with st.spinner("🤔 Đang suy nghĩ..."):
                        res = ai.generate(
                            full_prompt, 
                            model_type="flash", 
                            system_instruction=sys_instruction
                        )
                        
                        if res:
                            st.write(res)
                            
                            st.session_state.weaver_chat.append({
                                "role": "assistant", 
                                "content": res
                            })
                            
                            full_content = f"""
                            👤 USER: {prompt}

                            🤖 {persona}: {res}
                            """
                            
                            # ✅ GỌI HÀM MỚI
                            luu_lich_su(
                                loai="Tranh Biện Solo",
                                tieu_de=f"{persona} - {prompt[:50]}...",
                                noi_dung=full_content.strip()
                            )
                        else:
                            st.error("⚠️ AI không phản hồi. Vui lòng thử lại.")
        
        else:
            st.info("💡 Chọn 2-3 nhân vật để họ tự tranh luận.")
            
            participants = st.multiselect(
                "Chọn Hội Đồng Tranh Biện:", 
                list(DEBATE_PERSONAS.keys()), 
                default=[list(DEBATE_PERSONAS.keys())[0], list(DEBATE_PERSONAS.keys())[1]],
                max_selections=3,
                key="w_t3_multi_p"
            )
            
            topic = st.text_input(
                "Chủ đề tranh luận:", 
                placeholder="VD: Tiền có mua được hạnh phúc không?",
                key="w_t3_topic"
            )
            
            c_start, c_del = st.columns([1, 5])
            
            with c_start:
                start_btn = st.button(
                    "🔥 KHAI CHIẾN", 
                    key="w_t3_start", 
                    disabled=(len(participants) < 2 or not topic),
                    type="primary"
                )
            
            with c_del:
                if st.button("🗑️ Xóa Bàn", key="w_t3_multi_clr"):
                    st.session_state.weaver_chat = []
                    st.rerun()

            for msg in st.session_state.weaver_chat:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    st.info(content)
                else:
                    st.chat_message("assistant").write(content)
            
            if start_btn and topic and len(participants) >= 2:
                st.session_state.weaver_chat = []
                
                start_msg = f"📢 **CHỦ TỌA:** Khai mạc tranh luận về: *'{topic}'*"
                st.session_state.weaver_chat.append({"role": "system", "content": start_msg})
                st.info(start_msg)
                
                full_transcript = [start_msg]
                
                MAX_DEBATE_TIME = 90
                start_time = time.time()
                
                with st.status("🔥 Cuộc chiến đang diễn ra (tối đa 3 vòng)...") as status:
                    try:
                        for round_num in range(1, 4):
                            if time.time() - start_time > MAX_DEBATE_TIME:
                                st.warning("⏰ Đã hết thời gian tranh luận (90s). Kết thúc sớm.")
                                break
                            
                            status.update(label=f"🔄 Vòng {round_num}/3 đang diễn ra...")
                            
                            for i, p_name in enumerate(participants):
                                if time.time() - start_time > MAX_DEBATE_TIME:
                                    break
                                
                                if len(st.session_state.weaver_chat) > 1:
                                    recent_context = st.session_state.weaver_chat[-3:]
                                    context_str = "\n".join([
                                        f"- {m['content']}" 
                                        for m in recent_context 
                                        if m['role'] != 'system'
                                    ])
                                else:
                                    context_str = topic
                                
                                if round_num == 1:
                                    p_prompt = f"""
                                    CHỦ ĐỀ TRANH LUẬN: {topic}

                                    NHIỆM VỤ (Vòng 1 - Khai mạc): 
                                    Bạn là {p_name}. Hãy đưa ra quan điểm mở đầu của mình về chủ đề này.
                                    Nêu rõ lập trường và 2-3 lý lẽ chính (dưới 100 từ).
                                    """
                                else:
                                    p_prompt = f"""
                                    CHỦ ĐỀ: {topic}

                                    TÌNH HUỐNG HIỆN TẠI:
                                    {context_str}

                                    NHIỆM VỤ (Vòng {round_num} - Phản biện):
                                    Bạn là {p_name}. Hãy:
                                    1. Chỉ ra điểm yếu trong lập luận của đối thủ
                                    2. Củng cố quan điểm của mình
                                    3. Đưa ra thêm 1 ví dụ minh họa
                                    (Dưới 100 từ, súc tích)
                                    """
                                
                                try:
                                    res = ai.generate(
                                        p_prompt, 
                                        model_type="flash",
                                        system_instruction=DEBATE_PERSONAS[p_name]
                                    )
                                    
                                    if res:
                                        content_fmt = f"**{p_name}:** {res}"
                                        st.session_state.weaver_chat.append({
                                            "role": "assistant", 
                                            "content": content_fmt
                                        })
                                        full_transcript.append(content_fmt)
                                        
                                        with st.chat_message("assistant"):
                                            st.write(content_fmt)
                                        
                                        time.sleep(2)
                                    
                                except Exception as e:
                                    st.error(f"⚠️ Lỗi khi gọi AI cho {p_name}: {str(e)}")
                                    continue
                        
                        status.update(label="✅ Tranh luận kết thúc!", state="complete")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi nghiêm trọng: {e}")
                        status.update(label="❌ Tranh luận gặp lỗi", state="error")
                
                full_log = "\n\n".join(full_transcript)
                
                # ✅ GỌI HÀM MỚI
                luu_lich_su(
                    loai="Hội Đồng Tranh Biện",
                    tieu_de=f"Chủ đề: {topic}",
                    noi_dung=full_log
