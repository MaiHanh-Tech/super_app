import google.generativeai as genai
import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

class AI_Core:
    def __init__(self):
        try:
            api_key = st.secrets["api_keys"]["gemini_api_key"]
            genai.configure(api_key=api_key)
            
            # Khởi tạo các model
            self.flash = genai.GenerativeModel('gemini-2.5-flash')
            self.pro = genai.GenerativeModel('gemini-2.5-pro')
            # Thử thêm bản experimental nếu có
            self.exp = genai.GenerativeModel('gemini-2.5-flash-latest')
        except Exception as e:
            st.error(f"Lỗi API Key: {e}")

    def _get_safety(self):
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def generate(self, prompt, model_type="flash", system_instruction=None):
    """
    Chiến thuật tối ưu: Flash → Pro → Exp (chỉ khi thực sự cần)
    Không retry cùng model liên tiếp trong vòng 10s
    """
    
    # 1. Xây dựng Prompt
    full_prompt = prompt
    if system_instruction:
        full_prompt = f"SYSTEM INSTRUCTION:\n{system_instruction}\n\nUSER REQUEST:\n{prompt}"

    # 2. Chiến lược model dựa trên độ phức tạp
    if model_type == "pro":
        # Với task phức tạp: Pro → Flash (fallback) → Exp
        plan = [
            (self.pro, "Pro", 3),
            (self.flash, "Flash", 5),
            (self.exp, "Exp", 8)
        ]
    else:
        # Với task thường: Flash → Exp → Pro (cuối cùng mới dùng Pro tốn tiền)
        plan = [
            (self.flash, "Flash", 2),
            (self.exp, "Exp", 5),
            (self.pro, "Pro", 10)
        ]

    # 3. Thực thi với tracking lỗi
    last_errors = []
    
    for model, name, wait_time in plan:
        try:
            if not model: 
                continue
            
            response = model.generate_content(
                full_prompt, 
                safety_settings=self._get_safety()
            )
            
            if response.text:
                return response.text
                
        except ResourceExhausted as e:
            error_msg = f"{name}: Quota exhausted"
            last_errors.append(error_msg)
            time.sleep(wait_time)
            
        except (ServiceUnavailable, InternalServerError) as e:
            error_msg = f"{name}: Service error"
            last_errors.append(error_msg)
            time.sleep(wait_time * 0.5)  # Chờ ngắn hơn cho lỗi tạm thời
            
        except Exception as e:
            error_msg = f"{name}: {str(e)[:100]}"
            last_errors.append(error_msg)
            time.sleep(1)
                
    # Nếu tất cả đều fail
    return f"⚠️ Hệ thống quá tải:\n" + "\n".join(f"- {e}" for e in last_errors[-3:])
    
    @st.cache_data(ttl=3600)
    def analyze_static(_self, text, instruction):
        """Hàm dùng cho RAG - Có Cache"""
        # Với hàm này ta tạo instance mới để tránh conflict
        try:
            m = genai.GenerativeModel('gemini-2.5-flash')
            res = m.generate_content(f"{instruction}\n\n{text[:50000]}")
            return res.text
        except Exception as e:
            return f"Lỗi phân tích: {e}"
