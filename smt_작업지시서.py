# -*- coding: utf-8 -*-
r"""태블로에서 클릭 → 작업지시서 발행 (2026-10-02)

  태블로 「AI 판정 검증 센터」 의 신호등 칸이나 오판정 사례 줄에 마우스 → 도구 설명의 「📋 작업지시서 발행」 링크
  → Streamlit 페이지 (?wo=1&item=검사항목&band=처리구분[&box=박스ID]) → 이 모듈 issue()
  → 작업지시서 워드 + 사례 PDF (실제 검사 사진) → 슬랙 #smt-알림 → 발행 로그 (19_태블로/작업지시서_로그.csv)

  숫자는 전부 검증셋 CSV 에서 계산 (태블로 신호등과 같은 정의 : 같은 판정 구간 평균 + 3σ = 관리 한계)
  지시 사항은 정해진 규칙 문장이다 (원인을 단정하지 않음). 담당자 · 기한은 빈칸으로 둔다
"""
import os
import csv
import datetime

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

import smt_태블로_시연 as D
import smt_보고서_알림 as R

BAND_NAME = {'검사자 직접 확인': '우선 확인', '확인 필요': '일반 확인', 'AI 판정': '자동 판정'}
LOG = os.path.join(R.BASE, '작업지시서_로그.csv')


def cell_stats(df, item, band):
    b = df[df['처리구분'] == band]
    p = (b['맞음'] != '맞음').mean()
    n_bar = b.groupby('검사항목').size().mean()
    ucl = p + 3 * (p * (1 - p) / n_bar) ** 0.5
    c = b[b['검사항목'] == item]
    err = c[c['맞음'] != '맞음']
    rate = len(err) / len(c) if len(c) else 0
    r1, p1, u1 = round(rate * 100, 1), round(p * 100, 1), round(ucl * 100, 1)
    state = '위험' if r1 >= u1 else ('주의' if r1 > p1 else '양호')
    leak = int(((err['오류종류'] == 'FN') & (err['처리구분'] == 'AI 판정')).sum())
    fn = int((err['오류종류'] == 'FN').sum()) - leak
    fp = int((err['오류종류'] == 'FP').sum())
    return dict(n=len(c), m=len(err), rate=r1, mean=p1, ucl=u1, state=state, leak=leak, fn=fn, fp=fp, err=err)


def pick_cases(err, box=None, k=6):
    e = err.copy()
    e['순서'] = e.apply(lambda r: 0 if (r['오류종류'] == 'FN' and r['처리구분'] == 'AI 판정') else (1 if r['오류종류'] == 'FN' else 2), axis=1)
    e = e.sort_values(['순서', '확신도'], ascending=[True, False])
    if box and box in set(e['박스ID']):
        e = pd.concat([e[e['박스ID'] == box], e[e['박스ID'] != box]])
    return e.head(k)


def instructions(item, bname, s):
    out = [f'「{item} · {bname}」 박스는 다음 검증 때까지 판정 구간과 관계없이 사람 확인으로 돌린다.',
           f'아래 사례 사진의 박스를 검사자가 다시 판정하고 결과를 검사 기록에 남긴다.',
           '재판정 결과와 사례 사진을 모델 담당에게 재학습 자료로 전달한다.',
           f'조치 후 다음 검증에서 이 칸의 오판정률이 관리 한계 {s["ucl"]}% 아래로 내려오는지 확인한다.']
    if s['leak']:
        out.insert(0, f'유출 {s["leak"]}건이 있다. 해당 박스가 들어간 제품의 출하 여부를 먼저 확인한다.')
    return out


def resolve(df, item, band, box=None):
    """선택에 빠진 값을 채운다 : 박스 ID 만 → 그 박스의 항목 · 구간 / 항목만 (항목별 막대) → 그 항목에서 가장 나쁜 구간"""
    if box and not (item and band):
        hit = df[df['박스ID'] == box]
        if len(hit):
            item, band = hit['검사항목'].iloc[0], hit['처리구분'].iloc[0]
    if item and not band:
        rank = {'위험': 2, '주의': 1, '양호': 0}
        cands = [(b_, cell_stats(df, item, b_)) for b_ in df['처리구분'].dropna().unique()]
        cands = [(b_, s) for b_, s in cands if s['n']]
        if cands:
            band = max(cands, key=lambda x: (rank[x[1]['state']], x[1]['rate'] - x[1]['ucl']))[0]
    return item, band


def load(applied=True):
    """태블로 화면과 같은 범위 : 「새 검사 결과 반영」 전에는 새 검사 묶음(촬영 D.BATCH)을 뺀다"""
    df = pd.read_csv(R.CSV, encoding='utf-8')
    return df if applied else df[df['시간대'] != D.BATCH]


def build(item, band, box=None, notify=True, applied=True):
    df = load(applied)
    scope = f'검증셋 {len(df):,}박스' + ('' if applied else f' (새 검사 묶음 {D.BATCH} 반영 전)')
    item, band = resolve(df, item, band, box)
    bname = BAND_NAME.get(band, band)
    s = cell_stats(df, item, band)
    now = datetime.datetime.now()
    wo = f'WO-{now:%Y%m%d-%H%M%S}'
    os.makedirs(R.OUT_DIR, exist_ok=True)
    cases = pick_cases(s['err'], box)
    sheet = D.leak_sheet(cases, os.path.join(R.OUT_DIR, f'{wo}_사례.png'),
                         f'{wo} · {item} · {bname} 오판정 사례 {len(cases)}건 (전체 {s["m"]}건 중)',
                         sub='표시한 박스 = AI 판정이 정답과 다른 부위 (유출 → 정상 오분류 → 과검 순, 같은 유형은 확신하고 틀린 순)')
    pdf = sheet[:-4] + '.pdf'

    doc = Document()
    doc.styles['Normal'].font.name, doc.styles['Normal'].font.size = 'Pretendard', Pt(10.5)
    doc.add_heading('작업지시서', level=1)
    t = doc.add_table(rows=0, cols=2)
    t.style = 'Light Grid Accent 1'
    rows = [('지시서 번호', wo), ('발행 시각', now.strftime('%Y-%m-%d %H:%M')),
            ('발행 근거', f'태블로 「AI 판정 검증 센터」 판정 구간별 오판정률 (신호등) · {scope}'),
            ('대상', f'검사 항목 {item} · 판정 구간 {bname}' + (f' · 선택 박스 {box}' if box else '')),
            ('상태', s['state']),
            ('근거 숫자', f'오판정률 {s["rate"]}% (박스 {s["n"]:,}개 중 {s["m"]:,}개) · 구간 평균 {s["mean"]}% · 관리 한계 {s["ucl"]}%'),
            ('오판정 내역', f'유출 {s["leak"]} · 정상 오분류 {s["fn"]} · 과검 {s["fp"]}'),
            ('담당자', ''), ('완료 기한', '')]
    for a, b_ in rows:
        c = t.add_row().cells
        c[0].text, c[1].text = a, b_
    doc.add_heading('지시 사항', level=2)
    for i, line in enumerate(instructions(item, bname, s), 1):
        doc.add_paragraph(f'{i}. {line}')
    doc.add_heading('오판정 사례 (실제 검사 사진)', level=2)
    doc.add_picture(sheet, width=Inches(6.3))
    doc.add_heading('확인', level=2)
    st_ = doc.add_table(rows=2, cols=3)
    st_.style = 'Table Grid'
    for j, h in enumerate(['발행', '조치', '확인']):
        st_.rows[0].cells[j].text = h
    doc.add_paragraph(f'숫자 기준 : AI Hub 공식 {scope} · 지시 사항은 정해진 규칙 문장이며 원인을 단정하지 않음')
    docx = os.path.join(R.OUT_DIR, f'{wo}_작업지시서.docx')
    doc.save(docx)

    msg = (f'<!channel> 📋 *작업지시서 발행* `{wo}`' + (f' · 선택 박스 `{box}`' if box else '') +
           f'\n대상 : *{item} · {bname}* · 상태 *{s["state"]}*\n'
           f'오판정률 {s["rate"]}% (박스 {s["n"]:,}개 중 {s["m"]:,}개) · 구간 평균 {s["mean"]}% · 관리 한계 {s["ucl"]}%\n'
           f'유출 {s["leak"]} · 정상 오분류 {s["fn"]} · 과검 {s["fp"]} · 첨부 : 사례 PDF (실제 사진) · 작업지시서 워드')
    sent = D.send_slack_bot(msg, [pdf, docx]) if notify else False

    new = not os.path.exists(LOG)
    with open(LOG, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['발행 시각', '지시서 번호', '검사 항목', '판정 구간', '선택 박스', '상태', '오판정률(%)', '관리 한계(%)', '슬랙'])
        w.writerow([now.strftime('%Y-%m-%d %H:%M:%S'), wo, item, bname, box or '', s['state'], s['rate'], s['ucl'], '보냄' if sent else '안 보냄'])
    return dict(wo=wo, item=item, band=bname, box=box, scope=scope, stats={k: v for k, v in s.items() if k != 'err'},
                pdf=pdf, png=sheet, docx=docx, slack=sent, msg=msg)


def send(r):
    """팝업의 「📤 슬랙으로 보내기」 : 발행해 둔 작업지시서를 슬랙으로 보내고 발행 로그의 슬랙 칸을 「보냄」 으로 고친다"""
    ok = D.send_slack_bot(r['msg'], [r['pdf'], r['docx']])
    if ok and os.path.exists(LOG):
        lg = pd.read_csv(LOG, encoding='utf-8-sig', dtype=str)
        lg.loc[lg['지시서 번호'] == r['wo'], '슬랙'] = '보냄'
        lg.to_csv(LOG, index=False, encoding='utf-8-sig')
    return ok


if __name__ == '__main__':
    import sys
    r = build(sys.argv[1] if len(sys.argv) > 1 else '일어섬', sys.argv[2] if len(sys.argv) > 2 else 'AI 판정',
              notify='--slack' in sys.argv)
    print(r['wo'], r['stats'], os.path.basename(r['pdf']), os.path.basename(r['docx']), '슬랙', r['slack'])
