# -*- coding: utf-8 -*-
"""Streamlit 페이지 : 태블로 Cloud 대시보드 임베딩 + 작업지시서 팝업 + 유출 경보 (2026-10-03)

  화면 흐름
    · 태블로에서 신호등 칸 · 오판정 사례 줄 · 선택한 사례를 고르면 → 위에 「📋 작업지시서 발행」 버튼
      → 누르면 같은 화면에 팝업 : 작업지시서 카드 + 사례 PDF · 작업지시서(워드) + 발행 로그
      → 팝업의 「📤 슬랙으로 보내기」 를 눌러야 슬랙 @channel 로 감 (잘못 누르거나 리허설 때 알림이 안 가게)
    · ?wo=1&item=..&band=..&box=.. 로 열면 작업지시서 페이지 (태블로 도구 설명 링크 · 예비용)

  재빈님 앱에 넣는 법
    1) 이 파일 · smt_tableau_component/ 폴더 · smt_태블로_시연.py · smt_작업지시서.py · smt_보고서_알림.py 를 앱 폴더에 복사
       data/검사상태_박스단위.csv 도 같이 (또는 SMT_DATA_DIR 환경 변수로 폴더 지정)
    2) requirements : tableauserverclient slack_sdk openai python-docx pandas requests pillow
    3) .streamlit/secrets.toml (값은 윤정수에게 받아 직접 넣기 · 깃허브에 올리지 말 것)
         TABLEAU_PAT_NAME / TABLEAU_PAT_SECRET / SLACK_BOT_TOKEN / SLACK_CHANNEL / OPENAI_API_KEY
    4) 앱의 태블로 페이지에서 :  import smt_태블로_시연_페이지 as T ;  T.page()
  혼자 실행 : streamlit run smt_태블로_시연_페이지.py
  태블로 Cloud 화면은 로그인한 사람만 보인다 → 발표 PC 브라우저에서 Tableau Cloud 에 먼저 로그인해 둘 것
"""
import os
import html

import streamlit as st
import streamlit.components.v1 as components

import smt_태블로_시연 as D

VIEW = f'{D.SERVER}/t/{D.SITE}/views/SMT_/AI'
# 화면에 띄울 태블로 : public (로그인 없음 · 기본) / cloud (로그인 필요) · 환경 변수 SMT_EMBED 또는 주소 ?tb=cloud
# 숫자 확인 · 경보 · PDF 는 화면과 관계없이 늘 Tableau Cloud API 로 한다
PUBLIC_SERVER, PUBLIC_VIEW = 'https://public.tableau.com', 'https://public.tableau.com/views/SMT_/AI'


def embed_target():
    mode = (st.query_params.get('tb') or os.environ.get('SMT_EMBED') or 'public').lower()   # ?embed= 는 Streamlit 예약어
    return (PUBLIC_SERVER, PUBLIC_VIEW) if mode == 'public' else (D.SERVER, VIEW)
_HERE = os.path.dirname(os.path.abspath(__file__))
from smt_viz import viz as _viz      # 컴포넌트는 영어 이름 모듈에서 등록 (한글 모듈 이름이면 클라우드에서 404)


@st.cache_data
def hours():
    return D.base_hours()


def tableau(applied, key='viz'):
    """태블로 임베딩 · 대시보드에서 마크를 고르면 {sheet, item, band, box} 를 돌려준다"""
    server, src = embed_target()
    return _viz(src=src, server=server, width=1600, height=1000,
                filter_field=None if applied else D.FILTER_FIELD, filter_values=None if applied else hours(),
                key=key, default=None)


# ───────── 작업지시서 카드 (팝업 · 예비 페이지 공용) ─────────
def issue(item, band, box, dry=False):
    import smt_작업지시서 as W
    applied = st.session_state.get('applied', True)     # 태블로 화면과 같은 범위로 계산
    key = f'wo::{item}::{band}::{box}::{applied}'
    if key not in st.session_state:                      # 다시 그려도 두 번 발행되지 않게
        with st.spinner('작업지시서를 만드는 중…'):            # 슬랙은 팝업의 「📤 슬랙으로 보내기」 를 눌렀을 때만
            st.session_state[key] = W.build(item, band, box, notify=False, applied=applied)
    return st.session_state[key]


def card(r, dry=False):
    import smt_작업지시서 as W
    s = r['stats']
    col = {'위험': ('#c0392b', '#fdecea'), '주의': ('#b7791f', '#fbf1d9'), '양호': ('#2f7d4f', '#e8f3ec')}[s['state']]
    top = st.container()                                  # 카드는 보내기 버튼을 처리한 뒤에 채운다 (보냄 표시가 바로 바뀌게)
    st.write('')
    d1, d2, d3 = st.columns([1, 1, 1.2])
    d1.download_button('📄 사례 PDF', open(r['pdf'], 'rb').read(), file_name=os.path.basename(r['pdf']),
                       mime='application/pdf', use_container_width=True, key='dl_pdf_' + r['wo'])
    d2.download_button('📝 작업지시서(워드)', open(r['docx'], 'rb').read(), file_name=os.path.basename(r['docx']),
                       mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                       use_container_width=True, key='dl_doc_' + r['wo'])
    if d3.button('✅ 슬랙으로 보냄' if r['slack'] else '📤 슬랙으로 보내기 (@channel)', type='primary',
                 disabled=r['slack'] in (True, 'dry'), use_container_width=True, key='send_' + r['wo']):
        if dry:
            r['slack'] = 'dry'
        else:
            with st.spinner('슬랙으로 보내는 중…'):
                try:
                    r['slack'] = W.send(r) or 'fail'
                except Exception as e:                    # 토큰 · 채널 문제로 앱이 멈추지 않게 (키 값은 화면에 안 나옴)
                    r['slack'] = 'fail'
                    st.error(f'슬랙 보내기 실패 : {type(e).__name__} · {str(e)[:120]}')
    sent = {True: '✅ 슬랙 #smt-알림 으로 보냄 (@channel · 사례 PDF · 작업지시서 워드)', 'dry': '시험 모드라 슬랙은 보내지 않음',
            'fail': '⚠ 슬랙 보내기 실패 (SLACK_BOT_TOKEN · SLACK_CHANNEL 확인)'}.get(r['slack'], '아직 슬랙으로 보내지 않음 → 「📤 슬랙으로 보내기」')
    tile = lambda lab, val, sub='': (
        f"<div style='flex:1;background:#f5f7fa;border-radius:10px;padding:12px 14px'>"
        f"<div style='font-size:13px;color:#6b7280'>{lab}</div>"
        f"<div style='font-size:26px;font-weight:700;color:#111;margin-top:2px'>{val}</div>"
        f"<div style='font-size:12px;color:#6b7280;margin-top:2px'>{sub}</div></div>")
    top.markdown(f"""
<div style="border:1px solid #e5e7eb;border-left:6px solid {col[0]};border-radius:12px;padding:18px 20px;background:#fff">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span style="background:{col[1]};color:{col[0]};font-weight:700;border-radius:999px;padding:3px 12px;font-size:14px">● {s['state']}</span>
    <span style="font-size:22px;font-weight:700;color:#111">{html.escape(r['item'])} · {html.escape(r['band'])}</span>
    <span style="margin-left:auto;font-family:monospace;color:#374151;font-size:14px">{r['wo']}</span>
  </div>
  <div style="font-size:14px;color:#374151;margin-top:6px">{'선택 박스 ' + html.escape(r['box']) + ' · ' if r.get('box') else ''}발행 근거 : 태블로 「AI 판정 검증 센터」 판정 구간별 오판정률</div>
  <div style="display:flex;gap:10px;margin-top:14px">
    {tile('오판정률', f"{s['rate']}%", f"박스 {s['n']:,}개 중 {s['m']:,}개")}
    {tile('구간 평균', f"{s['mean']}%", '같은 판정 구간 전체')}
    {tile('관리 한계', f"{s['ucl']}%", '구간 평균 + 3σ')}
    {tile('오판정 내역', f"{s['leak'] + s['fn'] + s['fp']}건", f"유출 {s['leak']} · 정상 오분류 {s['fn']} · 과검 {s['fp']}")}
  </div>
  <div style="font-size:13px;color:#6b7280;margin-top:12px">{sent} · 발행 로그에 기록함</div>
</div>""", unsafe_allow_html=True)
    t1, t2 = st.tabs(['📝 작업지시서 (워드 미리보기)', '📄 사례 PDF (실제 검사 사진)'])
    with t1:
        docx_preview(r['docx'], r['png'])
    with t2:
        st.caption('PDF 와 같은 내용 · 표시한 박스 = AI 판정이 정답과 다른 부위')
        st.image(r['png'], use_container_width=True)


def docx_preview(path, png):
    """워드 파일을 화면에서 바로 읽게 HTML 로 그린다 (워드 · 한글 프로그램이 없는 PC · 배포 서버에서도 됨)
       문단 · 제목 · 표는 HTML, 그림이 든 문단 자리에는 사례 그림(png)을 넣는다"""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(path)
    esc = lambda t: html.escape(t).replace('\n', '<br>')
    box = "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:26px 32px;color:#111;font-size:14px;line-height:1.65'>"
    buf = []

    def flush():
        if buf:
            st.markdown(box + ''.join(buf) + '</div>', unsafe_allow_html=True)
            buf.clear()
    for el in doc.element.body.iterchildren():
        tag = el.tag.split('}')[-1]
        if tag == 'p':
            p = Paragraph(el, doc)
            if el.xpath('.//*[local-name()="drawing"]'):
                flush()
                st.image(png, use_container_width=True)
                continue
            t, sty = p.text.strip(), (p.style.name or '')
            if not t:
                continue
            if sty.startswith('Heading 1') or sty == 'Title':
                buf.append(f"<div style='font-size:22px;font-weight:800;margin:0 0 12px'>{esc(t)}</div>")
            elif sty.startswith('Heading'):
                buf.append(f"<div style='font-size:16px;font-weight:700;color:#213c60;margin:18px 0 6px;border-bottom:2px solid #c9d8ec;padding-bottom:3px'>{esc(t)}</div>")
            else:
                buf.append(f"<div style='margin:3px 0'>{esc(t)}</div>")
        elif tag == 'tbl':
            rows = Table(el, doc).rows
            cells = [[c.text for c in row.cells] for row in rows]
            if cells and len(cells[0]) == 2:                  # 항목 · 내용 표 (지시서 머리)
                tr = ''.join(f"<tr><th style='background:#eaf2fb;color:#213c60;text-align:left;padding:6px 10px;border:1px solid #c9d8ec;width:22%;white-space:nowrap'>{esc(a)}</th>"
                             f"<td style='padding:6px 10px;border:1px solid #c9d8ec'>{esc(b) or '&nbsp;'}</td></tr>" for a, b in cells)
            else:                                             # 확인 칸 (발행 · 조치 · 확인)
                tr = ''.join('<tr>' + ''.join(f"<td style='padding:6px 10px;border:1px solid #c9d8ec;text-align:center;height:{30 if i else 24}px;"
                                              f"{'background:#eaf2fb;font-weight:700;color:#213c60' if i == 0 else ''}'>{esc(c) or '&nbsp;'}</td>" for c in row) + '</tr>'
                             for i, row in enumerate(cells))
            buf.append(f"<table style='border-collapse:collapse;width:100%;margin:6px 0 10px'>{tr}</table>")
    flush()


def log_table(n=10):
    import smt_작업지시서 as W
    if os.path.exists(W.LOG):
        import pandas as pd
        st.markdown('**발행 로그 (최근 10건)**')
        st.dataframe(pd.read_csv(W.LOG, encoding='utf-8-sig').fillna('').tail(n).iloc[::-1],
                     hide_index=True, use_container_width=True)


@st.dialog('📋 작업지시서 발행', width='large')
def wo_dialog(item, band, box):
    card(issue(item, band, box), dry=bool(st.query_params.get('dry')))
    log_table(5)


def work_order_page(qp):
    """예비용 : 태블로 도구 설명 링크로 새 탭에서 열린 경우 (?wo=1&item=..&band=..&box=..)"""
    st.subheader('📋 작업지시서 발행')
    card(issue(qp.get('item', ''), qp.get('band', ''), qp.get('box') or None), dry=bool(qp.get('dry')))
    log_table()


# ───────── 메인 ─────────
def page():
    qp = st.query_params
    if qp.get('wo'):
        work_order_page(qp)
        return
    st.session_state.applied = True                       # 전체 검증셋 (새 검사 결과 반영 버튼은 뺌 · 경보는 재빈님 담당)
    st.subheader('AI 판정 검증 센터')
    st.caption('신호등 칸 · 오판정 사례 · 선택한 사례를 클릭하면 위에 「📋 작업지시서 발행」 버튼이 생겨요 '
               '→ 누르면 작업지시서 팝업 → 「📤 슬랙으로 보내기」 로 #smt-알림 에 알림 · 사례 PDF · 작업지시서 워드')

    bar = st.container()                                  # 선택 → 발행 버튼 자리 (태블로 위)
    sel = tableau(st.session_state.applied)
    if qp.get('sel_item'):                                # 시험용 : ?sel_item=일어섬&sel_band=AI 판정&dry=1 (태블로 선택 흉내)
        sel = {'item': qp.get('sel_item'), 'band': qp.get('sel_band'), 'box': qp.get('sel_box'), 't': 0}
    if sel and (sel.get('item') or sel.get('box')):
        import smt_작업지시서 as W
        if not (sel.get('item') and sel.get('band')):     # 박스 ID 만 → 항목 · 구간 / 항목만 (항목별 막대) → 가장 나쁜 구간
            it, bd = W.resolve(W.load(st.session_state.get('applied', True)), sel.get('item'), sel.get('band'), sel.get('box'))
            sel = {**sel, 'item': it, 'band': bd}
        label = (f"{sel.get('item') or ''} · {W.BAND_NAME.get(sel.get('band'), sel.get('band') or '')}"
                 + (f" · {sel['box']}" if sel.get('box') else ''))
        b1, b2 = bar.columns([3, 1.2])
        b1.info(f'선택 : {label}')
        if sel.get('open') and st.session_state.get('wo_link_t') != sel.get('t'):   # 도구 설명 「📋 작업지시서 발행」 링크 → 바로 팝업 (한 번만)
            st.session_state.wo_link_t = sel.get('t')
            wo_dialog(sel.get('item'), sel.get('band'), sel.get('box'))
        elif b2.button('📋 작업지시서 발행', type='primary', use_container_width=True, key=f"wo_{sel.get('t')}"):
            wo_dialog(sel.get('item'), sel.get('band'), sel.get('box'))


if __name__ == '__main__':          # streamlit run 으로 혼자 열 때 (재빈님 앱에서는 T.page() 를 부른다)
    st.set_page_config(page_title='SMT 시연 · AI 판정 검증 센터', layout='wide')
    page()
