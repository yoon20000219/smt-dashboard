# -*- coding: utf-8 -*-
"""
SMT 공정 총 관리 시스템 (총공정책임자용)
========================================
Tableau 「AI 판정 검증 센터」(퍼블릭) 를 Embedding API 로 임베딩하고,
대시보드에서 고른 오판정 사례 · 신호등 칸으로 작업지시서(워드 · 사례 PDF)를 바로 발행한다.
슬랙은 팝업 안 「📤 슬랙으로 보내기」 를 눌렀을 때만 보낸다 (Secrets 의 SLACK_BOT_TOKEN · SLACK_CHANNEL).

화면 코드 = smt_태블로_시연_페이지.py (page()) · 작업지시서 = smt_작업지시서.py
시험 : 주소 끝에 ?dry=1 이면 슬랙을 보내지 않는다.
"""

import streamlit as st

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #0b0b0b; }
    section[data-testid="stSidebar"] { background-color: #f9f9f7; border-right: 1px solid rgba(11,11,11,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

import smt_태블로_시연_페이지 as T

T.page()
