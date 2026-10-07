# -*- coding: utf-8 -*-
"""
SMT 공정 관리 시스템 (진입점)
=============================
좌측 메뉴에서 "품질관리 시스템"(현장 관리자용)과 "총 관리 시스템"(총공정책임자용, Tableau 임베딩 + 작업지시서)을
전환합니다.

실행:
    streamlit run app.py

총 관리 시스템 주소 = <배포주소>/ai-center  (태블로 도구 설명 「📋 작업지시서 발행」 링크가 이 주소로 온다)
"""

import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SMT_DATA_DIR", os.path.join(HERE, "data"))   # AI 판정 검증 센터 데이터 = 이 저장소의 data/
sys.path.insert(0, HERE)

st.set_page_config(
    page_title="SMT 공정 관리 시스템",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* 좌측 메뉴 위 제목 */
    [data-testid="stSidebarNav"]::before {
        content: "SMT 공정 관리 시스템";
        display: block; font-size: 1.1rem; font-weight: 800; letter-spacing: .01em; color: #0b0b0b;
        padding: 6px 16px 14px 16px; margin-bottom: 10px; border-bottom: 2px solid #0b0b0b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

quality_page = st.Page(
    "pages/01_품질관리_시스템.py",
    title="품질관리 시스템",
    icon=":material/precision_manufacturing:",
    default=True,
)
overview_page = st.Page(
    "pages/02_총관리_시스템.py",
    title="총 관리 시스템",
    icon=":material/dashboard:",
    url_path="ai-center",
)

pg = st.navigation([quality_page, overview_page])
pg.run()
