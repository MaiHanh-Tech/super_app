import streamlit as st
from datetime import datetime

class AuthBlock:
    def __init__(self):
        # 1. Load Config từ Secrets
        self.admin_pass = st.secrets.get("admin_password", "")
        self.users_db = st.secrets.get("users", {})
        self.tiers = st.secrets.get("user_tiers", {})
        
        limits = st.secrets.get("usage_limits", {})
        self.default_limit = limits.get("default_daily_limit", 30000)
        self.premium_limit = limits.get("premium_daily_limit", 500000)

        # 2. Khởi tạo Session State
        if 'user_logged_in' not in st.session_state:
            st.session_state.user_logged_in = False
        if 'usage_tracking' not in st.session_state:
            st.session_state.usage_tracking = {}

    def login(self, password):
        """Xử lý đăng nhập, trả về True/False"""
        if not password: return False

        # A. Kiểm tra Admin
        if password == self.admin_pass:
            self._set_user_session("Admin", is_admin=True, is_vip=True)
            return True

        # B. Kiểm tra User thường
        # Duyệt qua dict users để tìm password khớp (Value -> Key)
        for username, stored_pass in self.users_db.items():
            if password == stored_pass:
                # Kiểm tra Tier
                tier = self.tiers.get(username, "default")
                is_vip = (tier == "premium")
                self._set_user_session(username, is_admin=False, is_vip=is_vip)
                return True
        
        return False

    def _set_user_session(self, username, is_admin, is_vip):
        st.session_state.user_logged_in = True
        st.session_state.current_user = username
        st.session_state.is_admin = is_admin
        st.session_state.is_vip = is_vip

    def check_quota_status(self):
        """Trả về: (đã_dùng, giới_hạn, được_phép_không)"""
        if st.session_state.get('is_vip', False): 
            return 0, float('inf'), True # VIP không giới hạn

        username = st.session_state.get('current_user', 'guest')
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Init tracking nếu chưa có
        if username not in st.session_state.usage_tracking:
            st.session_state.usage_tracking[username] = {}
            
        current = st.session_state.usage_tracking[username].get(today, 0)
        limit = self.default_limit
        
        return current, limit, (current < limit)

    def track_usage(self, char_count):
        """Ghi nhận dung lượng sử dụng"""
        if st.session_state.get('is_vip', False): return

        username = st.session_state.get('current_user')
        if not username: return
        
        today = datetime.now().strftime("%Y-%m-%d")
        current = st.session_state.usage_tracking[username].get(today, 0)
        st.session_state.usage_tracking[username][today] = current + char_count
