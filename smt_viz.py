# -*- coding: utf-8 -*-
"""태블로 임베딩 컴포넌트 등록 (모듈 이름을 영어로 둔다)

컴포넌트 주소는 「모듈 이름.컴포넌트 이름」 으로 만들어진다. 모듈 이름이 한글이면
스트림릿 클라우드에서 그 주소가 404 가 나서 태블로가 안 뜬다 → 이 파일에서 등록한다.
"""
import os

import streamlit.components.v1 as components

_HERE = os.path.dirname(os.path.abspath(__file__))
viz = components.declare_component('smt_tableau_viz', path=os.path.join(_HERE, 'smt_tableau_component'))
