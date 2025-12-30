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

# ✅ IMPORT THƯ VIỆN SUPABASE
try:
    from supabase import create_client, Client
except ImportError:
    st.error("⚠️ Thiếu thư viện supabase. Hãy thêm 'supabase' vào requirements.txt")

# --- IMPORT CÁC META-BLOCKS ---
from ai_core import AI_Core
from voice_block import Voice_Engine
from prompts import DEBATE_PERSONAS, BOOK_ANALYSIS_PROMPT

# ==========================================
# ✅ CẤU HÌNH KẾT NỐI SUPABASE
# ==========================================
has_db = False
supabase = None

try:
    SUPA_URL = st.secrets["supabase"]["url"]
    SUPA_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPA_URL, SUPA_KEY)
    has_db = True
except Exception as e:
    pass

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
    # ... (Các ngôn ngữ khác giữ nguyên, rút gọn để tiết kiệm chỗ hiển thị)
}

def T(key):
    lang = st.session_state.get('weaver_lang', 'vi')
    # Fallback cơ bản nếu dict chưa đủ
    return TRANS.get('vi', {}).get(key, key)

# --- CÁC HÀM PHỤ TRỢ ---
@st.cache_resource
def load_models():
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
        model.max_seq_length = 128
        return model
    except: return None

def check_model_available():
    return load_models() is not None

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

# ==========================================
# ✅ CÁC HÀM TƯƠNG TÁC DB (ĐÃ SỬA TÊN BẢNG History_Logs)
# ==========================================

def luu_lich_su(loai, tieu_de, noi_dung):
    """Lưu log vào Supabase"""
    if not has_db: return
    user = st.session_state.get("current_user", "Unknown")
    data = {
        "type": loai,
        "title": tieu_de,
        "content": noi_dung,
        "user_name": user,
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral"
    }
    try:
        # ✅ FIX: Dùng tên bảng History_Logs (Viết Hoa)
        supabase.table("History_Logs").insert(data).execute()
    except Exception as e:
        print(f"Lỗi lưu log: {e}")

def tai_lich_su():
    """Tải log từ Supabase"""
    if not has_db: return []
    try:
        # ✅ FIX: Dùng tên bảng History_Logs (Viết Hoa)
        response = supabase.table("History_Logs").select("*").order("created_at", desc=True).limit(50).execute()
        raw_data = response.data
        formatted_data = []
        for item in raw_data:
            raw_time = item.get("created_at", "")
            clean_time = raw_time.replace("T", " ")[:19]
            formatted_data.append({
                "Time": clean_time,
                "Type": item.get("type"),
                "Title": item.get("title"),
                "Content": item.get("content"),
                "User": item.get("user_name"),
                "SentimentScore": item.get("sentiment_score", 0.0),
                "SentimentLabel": item.get("sentiment_label", "Neutral")
            })
        return formatted_data
    except: return []

# --- HÀM CHÍNH: RUN() ---
def run():
    ai = AI_Core()
    voice = Voice_Engine()
    
    with st.sidebar:
        st.markdown("---")
        lang_choice = st.selectbox("🌐 Ngôn ngữ", ["Tiếng Việt", "English", "中文"], key="weaver_lang_selector")
        if lang_choice == "Tiếng Việt": st.session_state.weaver_lang = 'vi'
        elif lang_choice == "English": st.session_state.weaver_lang = 'en'
        else: st.session_state.weaver_lang = 'zh'
    
    st.header(f"🧠 The Cognitive Weaver")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # === TAB 1: RAG & GRAPH & UPLOAD FILE ===
    with tab1:
        st.subheader(T("t1_header"))
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: file_excel = st.file_uploader(T("t1_up_excel"), type="xlsx", key="w_t1_ex")
        with c2: uploaded_files = st.file_uploader(T("t1_up_doc"), type=["pdf", "docx", "txt"], accept_multiple_files=True, key="w_t1_doc")
        with c3: 
            st.write(""); st.write("")
            btn_run = st.button(T("t1_btn"), type="primary", use_container_width=True)

        if btn_run and uploaded_files:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            vec = load_models()
            db, df = None, None
            has_db_rag = False
            
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                    db = vec.encode([f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in df.iterrows()])
                    has_db_rag = True
                    st.success(T("t1_connect_ok").format(n=len(df)))
                except: st.error("Lỗi đọc Excel.")

            for file_idx, f in enumerate(uploaded_files):
                status_text.text(f"Đang xử lý file {file_idx+1}/{total_files}: {f.name}")
                progress_bar.progress((file_idx) / total_files)
                
                text = doc_file(f)
                link = ""
                if has_db_rag and vec:
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
                    luu_lich_su("Phân Tích Sách", f.name, res[:200])

                # ✅ [MỚI] UPLOAD FILE LÊN SUPABASE STORAGE
                if has_db:
                    try:
                        f.seek(0)
                        file_bytes = f.read()
                        folder = datetime.now().strftime("%Y_%m_%d")
                        path = f"{folder}/{f.name}"
                        supabase.storage.from_("book_files").upload(
                            path=path, file=file_bytes, 
                            file_options={"content-type": f.type, "x-upsert": "true"}
                        )
                        st.toast(f"💾 Đã lưu file '{f.name}' lên Cloud!", icon="☁️")
                    except Exception as e:
                        print(f"Upload error: {e}")

                progress_bar.progress((file_idx+1) / total_files)
            
            status_text.text("✅ Hoàn thành!")

        # VẼ GRAPH (Giữ nguyên)
        if file_excel:
            try:
                with st.expander(T("t1_graph_title"), expanded=False):
                    vec = load_models()
                    if "book_embs" not in st.session_state: st.session_state.book_embs = vec.encode(df["Tên sách"].tolist())
                    embs = st.session_state.book_embs
                    sim = cosine_similarity(embs)
                    nodes, edges = []; max_nodes = st.slider("Max Nodes:", 5, len(df), min(50, len(df))); threshold = st.slider("Threshold:", 0.0, 1.0, 0.45)
                    for i in range(max_nodes):
                        nodes.append(Node(id=str(i), label=df.iloc[i]["Tên sách"], size=20, color="#FFD166"))
                        for j in range(i+1, max_nodes):
                            if sim[i,j]>threshold: edges.append(Edge(source=str(i), target=str(j), color="#118AB2"))
                    agraph(nodes, edges, Config(width=900, height=600, directed=False, physics=True, collapsible=False))
            except: pass

    # === TAB 2: DỊCH GIẢ ===
    with tab2:
        st.subheader(T("t2_header"))
        txt = st.text_area(T("t2_input"), height=150, key="w_t2_inp")
        c_l, c_s, c_b = st.columns([1,1,1])
        with c_l: target_lang = st.selectbox(T("t2_target"), ["Tiếng Việt", "English", "Chinese", "French", "Japanese"], key="w_t2_lang")
        with c_s: style = st.selectbox(T("t2_style"), ["Default", "Academic", "Literary", "Business"], key="w_t2_style")
        if st.button(T("t2_btn"), key="w_t2_btn") and txt:
            with st.spinner("..."):
                p = f"Translate to {target_lang}. Style: {style}. Text: {txt}"
                res = ai.generate(p, model_type="pro")
                st.markdown(res)
                luu_lich_su("Dịch Thuật", f"{target_lang}", txt[:50])

    # === TAB 3: ĐẤU TRƯỜNG ===
    with tab3:
        st.subheader(T("t3_header"))
        mode = st.radio("Mode:", ["👤 Solo", "⚔️ Multi-Agent"], horizontal=True, key="w_t3_mode")
        if "weaver_chat" not in st.session_state: st.session_state.weaver_chat = []

        if mode == "👤 Solo":
            c1, c2 = st.columns([3, 1])
            with c1: persona = st.selectbox(T("t3_persona_label"), list(DEBATE_PERSONAS.keys()), key="w_t3_solo_p")
            with c2: 
                if st.button(T("t3_clear"), key="w_t3_clr"): st.session_state.weaver_chat = []; st.rerun()
            for msg in st.session_state.weaver_chat: st.chat_message(msg["role"]).write(msg["content"])
            if prompt := st.chat_input(T("t3_input")):
                st.chat_message("user").write(prompt)
                st.session_state.weaver_chat.append({"role": "user", "content": prompt})
                recent_history = st.session_state.weaver_chat[-10:]
                context_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent_history])
                full_prompt = f"LỊCH SỬ:\n{context_text}\n\nNHIỆM VỤ: Trả lời USER."
                with st.chat_message("assistant"):
                    res = ai.generate(full_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[persona])
                    if res:
                        st.write(res)
                        st.session_state.weaver_chat.append({"role": "assistant", "content": res})
                        luu_lich_su("Tranh Biện Solo", f"{persona}...", f"Q: {prompt}\nA: {res}")
        else:
            # Multi-Agent logic
            participants = st.multiselect("Chọn Hội Đồng:", list(DEBATE_PERSONAS.keys()), default=[list(DEBATE_PERSONAS.keys())[0], list(DEBATE_PERSONAS.keys())[1]], max_selections=3)
            topic = st.text_input("Chủ đề:", key="w_t3_topic")
            if st.button("🔥 KHAI CHIẾN", disabled=(len(participants)<2 or not topic)):
                st.session_state.weaver_chat = []
                st.session_state.weaver_chat.append({"role": "system", "content": f"Chủ đề: {topic}"})
                full_transcript = []
                MAX_TIME = 90; start_time = time.time()
                with st.status("🔥 Đang tranh luận...") as status:
                    for round_num in range(1, 4):
                        if time.time() - start_time > MAX_TIME: break
                        for p_name in participants:
                            if time.time() - start_time > MAX_TIME: break
                            p_prompt = f"CHỦ ĐỀ: {topic}. Vòng {round_num}. Phản biện ngắn gọn."
                            try:
                                res = ai.generate(p_prompt, model_type="flash", system_instruction=DEBATE_PERSONAS[p_name])
                                if res:
                                    fmt = f"**{p_name}:** {res}"
                                    st.session_state.weaver_chat.append({"role": "assistant", "content": fmt})
                                    full_transcript.append(fmt)
                                    st.chat_message("assistant").write(fmt)
                                    time.sleep(2)
                            except: continue
                    status.update(label="Kết thúc!", state="complete")
                luu_lich_su("Hội Đồng Tranh Biện", topic, "\n".join(full_transcript))

    # === TAB 4: PHÒNG THU ===
    with tab4:
        st.subheader(T("t4_header"))
        inp_v = st.text_area("Text:", height=200); btn_v = st.button(T("t4_btn"))
        if btn_v and inp_v:
            path = voice.speak(inp_v)
            if path: st.audio(path)

    # === TAB 5: NHẬT KÝ & CÔNG CỤ CHUYỂN DỮ LIỆU ===
    with tab5:
        st.subheader("⏳ Nhật Ký & Phản Chiếu Tư Duy")
        if st.button("🔄 Tải lại", key="w_t5_refresh"):
            st.session_state.history_cloud = tai_lich_su()
            st.rerun()
        
        data = st.session_state.get("history_cloud", tai_lich_su())
        if data:
            df_h = pd.DataFrame(data)
            if "SentimentScore" in df_h.columns:
                try:
                    df_h["score"] = pd.to_numeric(df_h["SentimentScore"], errors='coerce').fillna(0)
                    fig = px.line(df_h, x="Time", y="score", markers=True, color_discrete_sequence=["#76FF03"])
                    st.plotly_chart(fig, use_container_width=True)
                except: pass
            
            for index, item in df_h.iterrows():
                t = str(item.get('Time', '')); tp = str(item.get('Type', '')); ti = str(item.get('Title', ''))
                with st.expander(f"{t} | {tp} | {ti}"): st.markdown(item.get('Content', ''))
        else:
            st.info("Chưa có dữ liệu.")

        # =======================================================
        # ✅ [TOOL V4] CÔNG CỤ CHUYỂN DỮ LIỆU "BẤT TỬ"
        # =======================================================
        st.divider()
        with st.expander("🛠️ CÔNG CỤ CHUYỂN NHÀ (V4 - Final Fix)", expanded=True):
            st.info("Tool V4: Tự sửa tên bảng History_Logs và lỗi số thập phân 0,95.")
            uploaded_csv = st.file_uploader("Upload CSV từ Google Sheet:", type=["csv"])
            
            if uploaded_csv and st.button("🚀 BẮT ĐẦU CHUYỂN"):
                df_old = pd.read_csv(uploaded_csv)
                df_old.columns = df_old.columns.str.strip()
                progress_bar = st.progress(0); success_count = 0; error_count = 0; errors_log = []
                
                for idx, row in df_old.iterrows():
                    try:
                        # 1. Fix ngày tháng
                        raw_time = str(row.get('Time', '')).strip()
                        clean_time = datetime.now().isoformat()
                        if raw_time and raw_time.lower() != 'nan':
                            try: clean_time = pd.to_datetime(raw_time).strftime('%Y-%m-%d %H:%M:%S')
                            except: pass
                        
                        # 2. Fix số liệu 0,95 -> 0.95
                        raw_score = str(row.get('SentimentScore', '0')).replace(',', '.')
                        try: final_score = float(raw_score)
                        except: final_score = 0.0

                        data = {
                            "created_at": clean_time,
                            "type": str(row.get('Type', 'General')),
                            "title": str(row.get('Title', 'No Title')),
                            "content": str(row.get('Content', '')),
                            "user_name": str(row.get('User', 'Imported')),
                            "sentiment_score": final_score,
                            "sentiment_label": str(row.get('SentimentLabel', 'Neutral'))
                        }
                        # 3. Gửi lên bảng History_Logs (Hoa)
                        supabase.table("History_Logs").insert(data).execute()
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        errors_log.append(f"Dòng {idx}: {e}")
                    progress_bar.progress((idx + 1) / len(df_old))
                
                st.success(f"✅ Xong: {success_count} dòng.")
                if error_count > 0: st.error(f"⚠️ Lỗi {error_count} dòng (xem chi tiết bên dưới).")
                if errors_log: st.write(errors_log)
                time.sleep(1); st.rerun()
