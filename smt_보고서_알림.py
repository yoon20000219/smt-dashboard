# -*- coding: utf-8 -*-
r"""AI 판정 검증 센터 보고서 + 알림 (2026-10-02)

  흐름 : 검증셋 CSV 로 숫자 계산 → 태블로 퍼블릭 화면 그림 받기 → OpenAI API 로 보고서 문장 → 워드 저장 → (선택) 슬랙 알림
  숫자는 전부 이 코드가 계산하고, AI 는 주어진 숫자로 문장만 쓴다 (없는 숫자를 만들지 않게)
  재빈님 Streamlit 에서는 make_report() 하나만 불러 쓰면 된다 (버튼 → 워드 내려받기 · 알림)

  필요한 것
    OpenAI API 키 : 환경 변수 OPENAI_API_KEY (없으면 정해진 틀 문장으로 대신 쓴다 · Claude 키 ANTHROPIC_API_KEY 도 받음)
    슬랙 알림     : 환경 변수 SLACK_WEBHOOK_URL (없으면 알림 생략)

  실행 : python smt_보고서_알림.py            → 19_태블로\보고서\ 에 워드 저장
         python smt_보고서_알림.py --slack    → 저장 + 슬랙 알림
"""
import io
import os
import sys
import datetime

import pandas as pd
import requests
from docx import Document
from docx.shared import Inches, Pt

# 데이터 폴더 : SMT_DATA_DIR 환경 변수 → 이 PC 의 19_태블로 → 스크립트 옆 data 폴더 (재빈님 앱에 넣을 때)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.environ.get('SMT_DATA_DIR'), r'C:\Users\qkwlx\OneDrive\Desktop\최종 프로젝트\SMT\19_태블로', os.path.join(_HERE, 'data')]
BASE = next((p for p in _CANDS if p and os.path.isdir(p)), os.path.join(_HERE, 'data'))
CSV = os.path.join(BASE, '검사상태_박스단위.csv')
OUT_DIR = os.path.join(BASE, '보고서')
TABLEAU_VIEW = 'https://public.tableau.com/views/SMT_/AI'
MODEL = 'claude-opus-5-5'                                    # ANTHROPIC_API_KEY 가 있을 때
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-5-mini')  # OPENAI_API_KEY 가 있을 때 (모델은 환경 변수로 바꿀 수 있음)
BANDS = [('검사자 직접 확인', '우선 확인'), ('확인 필요', '일반 확인'), ('AI 판정', '자동 판정')]


# ───────── 1. 숫자 (태블로 화면과 같은 정의) ─────────
def compute_kpi(csv=CSV):
    df = pd.read_csv(csv, encoding='utf-8')
    fn, fp = df['오류종류'] == 'FN', df['오류종류'] == 'FP'
    auto = df['처리구분'] == 'AI 판정'
    real_def = int((df['실제'] == '불량').sum())
    real_ok = int((df['실제'] == '정상').sum())
    leak = int((fn & auto).sum())
    k = {
        '검증셋 박스': len(df),
        '실제 불량': real_def,
        '불량으로 판정': int((df['오류종류'] == 'TP').sum()),
        '정상 오분류': int(fn.sum()),
        '사람 확인으로 넘어감': int((fn & ~auto).sum()),
        '최종 유출': leak,
        '과검': int(fp.sum()),
    }
    k['재현율'] = k['불량으로 판정'] / real_def
    k['유출률'] = leak / real_def
    k['PPM'] = leak / len(df) * 1e6
    k['과검률'] = k['과검'] / real_ok

    # 신호등 : 같은 판정 구간 안에서 비교 · 관리 한계 = 구간 평균 + 3σ (n = 그 구간 항목당 평균 박스 수)
    # 판정은 화면과 같게 소수 첫째 자리 % 로 반올림해 비교 (위험 = 관리 한계 이상 · 주의 = 평균 초과)
    risk, band_sum = [], []
    for raw, name in BANDS:
        b = df[df['처리구분'] == raw]
        p = (b['맞음'] != '맞음').mean()
        n_bar = b.groupby('검사항목').size().mean()
        ucl = p + 3 * (p * (1 - p) / n_bar) ** 0.5
        band_sum.append((name, len(b), round(p * 100, 1), round(ucl * 100, 1)))
        g = b.groupby('검사항목')['맞음'].agg(lambda s: (s != '맞음').mean())
        for item, r in g.items():
            if round(r * 100, 1) >= round(ucl * 100, 1):
                risk.append((name, item, round(r * 100, 1)))
    k['구간'] = band_sum
    k['위험 칸'] = sorted(risk, key=lambda t: -t[2])
    leak_df = df[fn & auto]
    k['유출 항목'] = leak_df['검사항목'].value_counts().to_dict()
    return k


# ───────── 2. 태블로 퍼블릭 화면 그림 ─────────
def fetch_tableau_png(view=TABLEAU_VIEW, timeout=60):
    r = requests.get(view + '.png', params={':showVizHome': 'no'}, timeout=timeout)
    r.raise_for_status()
    return r.content


# ───────── 3. 보고서 문장 (Claude API) ─────────
def facts_text(k, alert=None):
    lines = [
        f"검증셋 {k['검증셋 박스']:,}박스",
        f"실제 불량 {k['실제 불량']:,} · 불량으로 판정 {k['불량으로 판정']:,} (재현율 {k['재현율']*100:.1f}%)",
        f"정상 오분류 {k['정상 오분류']:,} 중 사람 확인으로 넘어감 {k['사람 확인으로 넘어감']:,} · 최종 유출 {k['최종 유출']:,} "
        f"(유출률 {k['유출률']*100:.2f}% · {k['PPM']:.0f} PPM)",
        f"과검 {k['과검']:,} (실제 정상의 {k['과검률']*100:.1f}%)",
        '판정 구간별 오판정률 (평균 / 관리 한계) : ' + ' · '.join(f'{n} {p}% / {u}% ({c:,}박스)' for n, c, p, u in k['구간']),
        '위험 칸 (관리 한계 이상) : ' + ' · '.join(f'{b} {i} {r}%' for b, i, r in k['위험 칸']),
        '유출 항목 : ' + ' · '.join(f'{i} {c}건' for i, c in k['유출 항목'].items()),
    ]
    if alert:
        lines.append('현장 경보 : ' + ' · '.join(f'{a}: {b}' for a, b in alert.items()))
    return '\n'.join(lines)


SYSTEM = """너는 SMT 공정 품질 엔지니어의 보고서를 쓰는 도우미다.
주어진 숫자만 쓴다. 새 숫자를 계산하거나 지어내지 않는다. 원인을 단정하지 않는다.
용어 : 「라인」이라는 말은 쓰지 않는다 (데이터를 모은 업체는 「수집처」이며, 우선 확인 · 일반 확인 · 자동 판정은 「판정 구간」이다). 「유출 추정」 대신 「유출」. 사람 확인 구간으로 간 불량은 검사자가 「잡았다」고 쓰지 말고 「사람 확인으로 넘어갔다」처럼 문장에 맞게 쓴다.
형식 : 한국어 존댓말(~습니다) · 세 단락, 각 단락은 「1. 요약 : 」 「2. 먼저 볼 곳 : 」 「3. 권고 조치 : 」 로 시작하는 한 줄 · 단락마다 2~3문장 · 긴 줄표 쓰지 않음."""


def fallback_text(k):
    top = ', '.join(f'{b} {i}({r}%)' for b, i, r in k['위험 칸'][:3])
    return (f"1. 요약 : 실제 불량 {k['실제 불량']:,}건 중 {k['불량으로 판정']:,}건을 불량으로 판정했습니다(재현율 {k['재현율']*100:.1f}%). "
            f"최종 유출은 {k['최종 유출']}건({k['PPM']:.0f} PPM), 과검은 {k['과검']:,}건입니다.\n"
            f"2. 먼저 볼 곳 : 관리 한계를 넘은 위험 칸은 {len(k['위험 칸'])}곳이며, 오판정률이 높은 곳은 {top}입니다.\n"
            f"3. 권고 조치 : 위험 칸에 해당하는 항목은 자동 판정 구간이라도 사람 확인으로 돌리는 것을 검토합니다.")


def write_text(k, alert=None):
    """OPENAI_API_KEY 가 있으면 OpenAI (팀 기본) · 없고 ANTHROPIC_API_KEY 가 있으면 Claude · 둘 다 없으면 틀 문장"""
    prompt = '다음 숫자로 보고서를 써 줘.\n\n' + facts_text(k, alert)
    if os.environ.get('OPENAI_API_KEY'):
        import openai
        client = openai.OpenAI()
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            )
        except openai.APIError as e:
            return fallback_text(k), f'틀 문장 (OpenAI 오류 : {e.__class__.__name__})'
        text = (resp.choices[0].message.content or '').strip()
        return (text or fallback_text(k)), f'OpenAI ({resp.model})'
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return fallback_text(k), '틀 문장 (API 키 없음)'
    import anthropic
    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=4000, system=SYSTEM,
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',
            messages=[{'role': 'user', 'content': prompt}],
        )
    except anthropic.APIError as e:
        return fallback_text(k), f'틀 문장 (API 오류 : {e.__class__.__name__})'
    if resp.stop_reason == 'refusal':
        return fallback_text(k), '틀 문장 (응답 거절)'
    text = ''.join(b.text for b in resp.content if b.type == 'text').strip()
    return (text or fallback_text(k)), f'Claude ({resp.model})'


# ───────── 4. 워드 ─────────
def build_docx(k, text, png=None, alert=None, img_title='대시보드 화면'):
    doc = Document()
    st = doc.styles['Normal']
    st.font.name, st.font.size = 'Pretendard', Pt(10.5)
    doc.add_heading('AI 판정 검증 보고서', level=1)
    doc.add_paragraph(datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '  ·  검증셋 기준 (공식 Validation)')
    if alert:
        doc.add_heading('현장 경보', level=2)
        for a, b in alert.items():
            doc.add_paragraph(f'{a} : {b}', style='List Bullet')
    doc.add_heading('핵심 숫자', level=2)
    t = doc.add_table(rows=0, cols=2)
    t.style = 'Light Grid Accent 1'
    for a, b in [('실제 불량', f"{k['실제 불량']:,}"), ('불량으로 판정', f"{k['불량으로 판정']:,} (재현율 {k['재현율']*100:.1f}%)"),
                 ('정상 오분류', f"{k['정상 오분류']:,}"), ('사람 확인으로 넘어감', f"{k['사람 확인으로 넘어감']:,}"),
                 ('최종 유출', f"{k['최종 유출']:,} (유출률 {k['유출률']*100:.2f}% · {k['PPM']:.0f} PPM)"),
                 ('과검', f"{k['과검']:,} (실제 정상의 {k['과검률']*100:.1f}%)")]:
        c = t.add_row().cells
        c[0].text, c[1].text = a, b
    doc.add_heading('위험 칸 (관리 한계 이상)', level=2)
    for b, i, r in k['위험 칸']:
        doc.add_paragraph(f'{b} · {i} · {r}%', style='List Bullet')
    doc.add_heading('해석', level=2)
    for para in text.split('\n'):
        if para.strip():
            doc.add_paragraph(para.strip())
    if png:
        doc.add_heading(img_title, level=2)
        doc.add_picture(io.BytesIO(png), width=Inches(6.3))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ───────── 5. 슬랙 알림 ─────────
def send_slack(text, webhook=None):
    webhook = webhook or os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook:
        return False
    r = requests.post(webhook, json={'text': text}, timeout=15)
    return r.status_code == 200


def slack_text(k, alert=None):
    head = '🚨 *SMT 현장 경보*' if alert else '📋 *SMT AI 판정 검증 보고서*'
    body = (f"최종 유출 {k['최종 유출']}건 ({k['PPM']:.0f} PPM) · 과검 {k['과검']:,}건 · 재현율 {k['재현율']*100:.1f}%\n"
            f"위험 칸 {len(k['위험 칸'])}곳 : " + ', '.join(f'{b} {i} {r}%' for b, i, r in k['위험 칸'][:5]))
    if alert:
        body = ' · '.join(f'{a}: {b}' for a, b in alert.items()) + '\n' + body
    return f'{head}\n{body}\n대시보드 : {TABLEAU_VIEW}'


# ───────── 한 번에 ─────────
def make_report(alert=None, notify=False, with_png=True):
    """alert 예 : {'시각': '09-03 16:34', '경보': 'EWMA + 단일점 이탈', '기여 센서': '습도 58% · 소음 26%', '점검 대상': '자재 보관 점검 (예시 대응안)'}
    돌려주는 값 : (워드 바이트, 문장 출처, 슬랙 전송 여부)"""
    k = compute_kpi()
    png = None
    if with_png:
        try:
            png = fetch_tableau_png()
        except requests.RequestException:
            png = None
    text, src = write_text(k, alert)
    docx = build_docx(k, text, png, alert)
    sent = send_slack(slack_text(k, alert)) if notify else False
    return docx, src, sent


if __name__ == '__main__':
    docx, src, sent = make_report(notify='--slack' in sys.argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"AI판정검증_보고서_{datetime.datetime.now():%Y%m%d_%H%M}.docx")
    open(p, 'wb').write(docx)
    print('저장', p)
    print('문장', src, '· 슬랙', '보냄' if sent else '안 보냄')
