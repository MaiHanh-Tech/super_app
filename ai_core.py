import google.generativeai as genai
import streamlit as st
import time
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError, InvalidArgument

class AI_Core:
    def __init__(self):
        self.api_ready = False
        try:
            # Kiểm tra key tồn tại trước khi lấy
            if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                api_key = st.secrets["api_keys"]["gemini_api_key"]
                genai.configure(api_key=api_key)
                self.api_ready = True
            else:
                st.error("⚠️ Chưa cấu hình API Key trong secrets.toml")
                return

            # ✅ Cấu hình Safety chung
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            # ✅ Cấu hình Generation Config tối ưu cho Gemini 2.5
            self.gen_config = genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=32768,  # 32K (2.5 hỗ trợ đến 64K)
                top_p=0.95,
                top_k=40
            )

        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo AI Core: {e}")

    def _get_model(self, model_name, system_instr=None):
        """
        Hàm helper để khởi tạo model với system instruction
        
        Returns:
            GenerativeModel instance hoặc None nếu lỗi
        """
        # ✅ TÊN MODEL CHÍNH XÁC CHO GEMINI 2.5
        valid_names = {
            "flash": "gemini-2.5-flash",         # Ổn định, nhanh
            "pro": "gemini-2.5-pro",             # Mạnh nhất
            "exp": "gemini-2.5-flash-latest"     # Experimental (có thể thay đổi)
        }
        
        target_name = valid_names.get(model_name, "gemini-2.5-flash")
        
        try:
            return genai.GenerativeModel(
                model_name=target_name,
                safety_settings=self.safety_settings,
                generation_config=self.gen_config,
                system_instruction=system_instr
            )
        except Exception as e:
            st.warning(f"⚠️ Không thể khởi tạo model {target_name}: {e}")
            return None

    def generate(self, prompt, model_type="flash", system_instruction=None):
        """
        Generate content với fallback strategy
        
        Args:
            prompt: User prompt
            model_type: "flash", "pro", hoặc "exp"
            system_instruction: System instruction (optional)
        
        Returns:
            str: Generated text hoặc error message
        """
        if not self.api_ready:
            return "⚠️ API Key chưa sẵn sàng."

        # ✅ Chiến lược fallback thông minh
        if model_type == "pro":
            # Task phức tạp: Pro → Flash → Exp
            plan = [("pro", "Pro", 4), ("flash", "Flash", 2), ("exp", "Exp", 5)]
        else:
            # Task thường: Flash → Exp → Pro (tiết kiệm chi phí)
            plan = [("flash", "Flash", 2), ("exp", "Exp", 3), ("pro", "Pro", 5)]

        last_errors = []
        quota_exhausted_count = 0  # ✅ Đếm số lần hết quota

        for m_type, m_name, base_wait_time in plan:
            try:
                # Khởi tạo model
                model = self._get_model(m_type, system_instr=system_instruction)
                if not model:
                    continue  # Skip nếu không khởi tạo được
                
                # ✅ Gọi API
                response = model.generate_content(prompt)
                
                # ✅ KIỂM TRA RESPONSE ĐẦY ĐỦ
                if response and hasattr(response, 'text') and response.text:
                    return response.text
                
                # ✅ XỬ LÝ CÁC TRƯỜNG HỢP ĐẶC BIỆT
                if response and hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    if hasattr(candidate, 'finish_reason'):
                        reason = candidate.finish_reason.name
                        
                        if reason == "SAFETY":
                            error_msg = f"{m_name}: Response bị chặn bởi Safety Filter"
                            last_errors.append(error_msg)
                            continue  # Thử model khác
                            
                        elif reason == "MAX_TOKENS":
                            error_msg = f"{m_name}: Response quá dài (vượt max_tokens)"
                            last_errors.append(error_msg)
                            # Thử model khác với context ngắn hơn
                            continue
                
                # Nếu không có text nhưng không có lỗi cụ thể
                error_msg = f"{m_name}: Response rỗng (unknown reason)"
                last_errors.append(error_msg)
                continue
            
            except ResourceExhausted:
                quota_exhausted_count += 1
                error_msg = f"{m_name}: Hết quota (429)"
                last_errors.append(error_msg)
                
                # ✅ EXPONENTIAL BACKOFF
                backoff = base_wait_time * (2 ** (quota_exhausted_count - 1))
                backoff = min(backoff, 30)  # Tối đa 30s
                
                time.sleep(backoff)
                
            except (ServiceUnavailable, InternalServerError) as e:
                error_msg = f"{m_name}: Lỗi server Google (5xx)"
                last_errors.append(error_msg)
                time.sleep(1)  # Retry nhanh cho lỗi tạm thời
            
            except InvalidArgument as e:
                # ✅ Lỗi input không nên retry
                return f"⚠️ Lỗi Input (prompt không hợp lệ): {str(e)[:200]}"
                
            except Exception as e:
                error_msg = f"{m_name}: {str(e)[:100]}"
                last_errors.append(error_msg)
                time.sleep(1)

        # ✅ TẤT CẢ MODEL ĐỀU FAIL
        error_summary = "\n".join(f"- {e}" for e in last_errors[-3:])  # Chỉ hiện 3 lỗi cuối
        return f"⚠️ Hệ thống quá tải hoặc lỗi nghiêm trọng:\n{error_summary}\n\n💡 Vui lòng thử lại sau 1-2 phút."

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def analyze_static(text, instruction):
        """
        ✅ Hàm phân tích tĩnh với cache (cho RAG)
        
        Static method để Streamlit cache đúng cách
        """
        try:
            # Lấy API key mỗi lần gọi (vì static không có self)
            api_key = st.secrets["api_keys"]["gemini_api_key"]
            genai.configure(api_key=api_key)
            
            # ✅ Dùng Flash cho RAG (nhanh + rẻ)
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=instruction,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # ✅ Giới hạn input để tránh lỗi token (2.5 có context 2M tokens nhưng vẫn nên giới hạn)
            max_chars = 200000  # ~50K tokens
            truncated_text = text[:max_chars]
            
            if len(text) > max_chars:
                st.warning(f"⚠️ Text quá dài. Chỉ phân tích {max_chars:,} ký tự đầu.")
            
            response = model.generate_content(truncated_text)
            
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                return "⚠️ Không có response từ AI"
                
        except Exception as e:
            return f"❌ Lỗi phân tích tĩnh: {str(e)[:200]}"
