import streamlit as st
from utils.authentication import logout

def show():
    # ------------------------------------------------------
    # CUSTOM CSS FOR STYLING & CARDS
    # ------------------------------------------------------
    st.markdown("""
        <style>
        /* Card container base style */
        .stmacs-card {
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 10px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.08);
        }
        
        .stmacs-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }

        /* Module Theme Styles */
        .card-live {
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
            border-left: 6px solid #0284c7;
        }
        .card-twin {
            background: linear-gradient(135deg, #fae8ff 0%, #fdf4ff 100%);
            border-left: 6px solid #c026d3;
        }
        .card-pred {
            background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
            border-left: 6px solid #16a34a;
        }
        .card-trends {
            background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
            border-left: 6px solid #d97706;
        }
        .card-faults {
            background: linear-gradient(135deg, #ffe4e6 0%, #fff1f2 100%);
            border-left: 6px solid #e11d48;
        }

        /* Typography & Icon styling inside cards */
        .card-icon {
            font-size: 1.5rem;
            margin-bottom: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.05);
        }

        /* High-contrast badge specifically for Predictive Heating icon */
        .pred-icon-badge {
            background: #16a34a !important;
            color: #ffffff !important;
            box-shadow: 0 2px 6px rgba(22, 163, 74, 0.3);
        }

        .card-title {
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 4px;
            color: #1e293b;
        }
        .card-caption {
            font-size: 0.88rem;
            color: #475569;
            line-height: 1.4;
            min-height: 48px;
        }

        /* Banner styling */
        .welcome-banner {
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border: 1px solid #fed7aa;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------
    # WELCOME BANNER
    # ------------------------------------------------------
    st.markdown("""
        <div class="welcome-banner">
            <h3 style="color: #ea580c; margin-top:0;">☀️ Smart Solar Thermal Monitoring and Control System</h3>
            <h1 style="color: #9a3412; margin-bottom: 10px;">Welcome back, Gorata 👋</h1>
            <p style="color: #7c2d12; font-size: 1.05rem; margin-bottom:0;">
                Centralized dashboard for real-time telemetry, Digital twin visualization, predictive machine learning models, and thermal hardware status.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🚀 System Modules")
    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # MAJOR MODULE CARDS (3x2 GRID)
    # ------------------------------------------------------
    
    # ROW 1
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="stmacs-card card-live">
                <div class="card-icon">📡</div>
                <div class="card-title">Live Monitoring</div>
                <div class="card-caption">Real-time collector temperatures, flow telemetry, and live sensor streams.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Live Monitoring", use_container_width=True, key="btn_live"):
            st.session_state.page = "Live Monitoring"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="stmacs-card card-twin">
                <div class="card-icon">🧩</div>
                <div class="card-title">Digital Twin</div>
                <div class="card-caption">Interactive 3D/schematic model of the 300L thermal storage setup.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Digital Twin", use_container_width=True, key="btn_twin"):
            st.session_state.page = "Digital Twin"
            st.rerun()

    with col3:
        st.markdown("""
            <div class="stmacs-card card-pred">
                <div class="card-icon pred-icon-badge">🤖</div>
                <div class="card-title">Predictive Heating</div>
                <div class="card-caption">Machine learning thermal forecasting and dynamic demand control.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Predictive Control", use_container_width=True, key="btn_pred"):
            st.session_state.page = "Predictive Control"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
            <div class="stmacs-card card-trends">
                <div class="card-icon">📈</div>
                <div class="card-title">Historical Trends</div>
                <div class="card-caption">Long-term performance analytics, solar efficiency ratios, and log history.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Historical Trends", use_container_width=True, key="btn_trends"):
            st.session_state.page = "Historical Trends"
            st.rerun()

    with col5:
        st.markdown("""
            <div class="stmacs-card card-faults">
                <div class="card-icon">⚠️</div>
                <div class="card-title">Fault Logs</div>
                <div class="card-caption">Diagnostic codes, system alerts, automated safeguards, and security events.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Fault Logs", use_container_width=True, key="btn_faults"):
            st.session_state.page = "Fault Logs"
            st.rerun()

    with col6:
        # Empty column to balance row 2 layout
        pass

    # ------------------------------------------------------
    # FOOTER UTILITIES (ABOUT, SETTINGS, LOGOUT)
    # ------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()

    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])

    with f_col1:
        if st.button("ℹ️ About Project", use_container_width=True, key="btn_about"):
            st.session_state.page = "About Project"
            st.rerun()

    with f_col2:
        if st.button("⚙️ Settings", use_container_width=True, key="btn_settings"):
            st.session_state.page = "Settings"
            st.rerun()

    with f_col3:
        if st.button("🚪 Logout", use_container_width=True, key="btn_logout"):
            logout()
            st.rerun()