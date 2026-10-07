# -*- coding: utf-8 -*-
r"""발표 시연 : 「새 검사 결과 반영」 → 태블로 Cloud 가 다시 계산 → 기준 확인 → 슬랙 알림 · 보고서 (2026-10-02)

  방식 : 데이터 원본은 건드리지 않는다 (원본을 바꾸면 별칭 · 색 · 시트가 깨짐 → 10-02 시도 후 되돌림)
         시작 상태 = 태블로 필터 「촬영 시각」 에서 새 검사 묶음(09-02 17시)을 뺀 화면
         반영      = 필터를 풀어 새 묶음까지 포함한 화면 (= 발표 숫자와 같은 최종 상태)
         값 · PDF 는 전부 Tableau Cloud REST API (tableauserverclient) 가 계산해서 돌려준 것
  데이터는 지어내지 않는다. 새 검사 묶음 = 검증셋의 실제 한 시간대 (09-02 17시 · 4,400박스 · 유출 9건)

  필요한 것 (값은 코드에 쓰지 않음 : 환경 변수 → Streamlit secrets → 윈도우 사용자 환경 변수 순서로 찾음)
    TABLEAU_PAT_NAME / TABLEAU_PAT_SECRET   태블로 개인용 액세스 토큰
    SLACK_BOT_TOKEN / SLACK_CHANNEL         슬랙 봇 (메시지 + PDF · 워드 첨부)
    OPENAI_API_KEY                          (선택) 보고서 해석 문장

  실행 : python smt_태블로_시연.py check | push
"""
import os
import re
import sys
import time
import datetime
try:
    import winreg                         # 윈도우에서만 (Streamlit Cloud 는 리눅스)
except ImportError:
    winreg = None

import pandas as pd
import tableauserverclient as TSC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smt_보고서_알림 as R  # noqa: E402

SERVER = 'https://prod-kr-a.online.tableau.com'
SITE = 'qkwlxm123-08aa4e10b7'
VIEW_COUNT = '최종 유출 수'
VIEW_DASH = 'AI 판정 검증 센터'
FILTER_FIELD = '촬영 시각'               # 태블로 표시 이름 (원본 열 : 시간대)
BATCH = '09-02 17시'                     # 시연용 「새 검사 묶음」


def uenv(n):
    v = os.environ.get(n)
    if v:
        return v
    try:
        import streamlit as st
        if n in st.secrets:
            return str(st.secrets[n])
    except Exception:
        pass
    if winreg:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                return winreg.QueryValueEx(k, n)[0]
        except OSError:
            pass
    return None


def load_full():
    return pd.read_csv(R.CSV, encoding='utf-8')


def base_hours():
    """시작 상태에 보이는 촬영 시각 목록 (새 묶음만 뺌)"""
    return sorted(h for h in load_full()['시간대'].unique() if h != BATCH)


# ───────── Tableau Cloud ─────────
def sign_in():
    auth = TSC.PersonalAccessTokenAuth(uenv('TABLEAU_PAT_NAME'), uenv('TABLEAU_PAT_SECRET'), site_id=SITE)
    srv = TSC.Server(SERVER, use_server_version=True)
    srv.auth.sign_in(auth)
    return srv


def find_view(srv, name):
    for v in TSC.Pager(srv.views):
        if v.name == name and v.content_url.startswith('SMT_'):
            return v
    raise RuntimeError('뷰 없음 : ' + name)


def read_count(srv, hours=None):
    """태블로 「최종 유출 수」 값 · hours 를 주면 그 촬영 시각만 보이게 거른 값"""
    v = find_view(srv, VIEW_COUNT)
    opt = TSC.CSVRequestOptions(maxage=1)
    if hours:
        opt = opt.vf(FILTER_FIELD, ','.join(hours))
    srv.views.populate_csv(v, opt)
    lines = b''.join(v.csv).decode('utf-8-sig').strip().splitlines()
    return int(float(lines[-1].replace(',', '')))


def dash_pdf(srv, path):
    v = find_view(srv, VIEW_DASH)
    opt = TSC.PDFRequestOptions(page_type=TSC.PDFRequestOptions.PageType.A4,
                                orientation=TSC.PDFRequestOptions.Orientation.Landscape, maxage=1)
    srv.views.populate_pdf(v, opt)
    open(path, 'wb').write(v.pdf)
    return path


def dash_png(srv, path):
    """태블로 Cloud 가 그린 대시보드 그림 (슬랙에서 바로 펼쳐 보임)"""
    v = find_view(srv, VIEW_DASH)
    srv.views.populate_image(v, TSC.ImageRequestOptions(imageresolution=TSC.ImageRequestOptions.Resolution.High, maxage=1))
    open(path, 'wb').write(v.image)
    return path


ORIG_DIRS = [r'C:\smt\Validation', r'C:\smt\Training']
CROP_DIR = os.path.join(R.BASE, '_사례이미지')


def _font(size, bold=False):
    """한글 글꼴 : 묶음 안 fonts/ (배포 서버용) → 이 PC 에 설치된 프리텐다드 → 맑은 고딕"""
    from PIL import ImageFont
    name = 'Pretendard-' + ('Bold' if bold else 'Regular') + '.otf'
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', name),
              os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\Windows\Fonts' + '\\' + name),
              r'C:\Windows\Fonts\malgunbd.ttf' if bold else r'C:\Windows\Fonts\malgun.ttf']:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _photo(row):
    """원본 검사 사진에 그 박스를 빨갛게 그린 그림 · 원본이 없으면 사례 크롭 그림"""
    from PIL import Image, ImageDraw
    import glob
    for d in ORIG_DIRS:
        hit = glob.glob(os.path.join(d, '**', row['이미지'] + '.jpg'), recursive=True)
        if hit:
            im = Image.open(hit[0]).convert('RGB')
            dr = ImageDraw.Draw(im)
            x, y, w, h = float(row['x']), float(row['y']), float(row['w']), float(row['h'])
            for k in range(4):
                dr.rectangle((x - k, y - k, x + w + k, y + h + k), outline='#e02020')
            return im
    p = os.path.join(CROP_DIR, row['박스ID'] + '.jpg')
    return Image.open(p).convert('RGB') if os.path.exists(p) else Image.new('RGB', (512, 512), '#dddddd')


def leak_sheet(leak, path, title, sub='표시한 박스 = AI 가 「정상」 이라 확신해(자동 판정) 사람 확인 없이 넘어간 불량 부위'):
    """사례 시트 : 실제 검사 사진 + 빨간 박스 + 박스 ID · 항목 · 수집처 · 불량 확률 (PNG + 같은 내용 PDF)"""
    from PIL import Image, ImageDraw
    n = len(leak)
    cols = 3 if n > 4 else max(n, 1)
    rows = (n + cols - 1) // cols
    T, P, CAP = 360, 24, 92                     # 사진 한 칸 · 여백 · 글자 줄
    W = cols * T + (cols + 1) * P
    H = 120 + rows * (T + CAP + P) + P
    sheet = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((P, 26), title, font=_font(30, True), fill='#111111')
    d.text((P, 72), sub, font=_font(19), fill='#c0392b')
    for i, (_, r) in enumerate(leak.iterrows()):
        cx, cy = P + (i % cols) * (T + P), 120 + (i // cols) * (T + CAP + P)
        sheet.paste(_photo(r).resize((T, T)), (cx, cy))
        d.rectangle((cx, cy, cx + T - 1, cy + T - 1), outline='#c9d8ec', width=2)
        d.text((cx, cy + T + 8), f"{r['박스ID']}  ·  {r['검사항목']}", font=_font(21, True), fill='#111111')
        d.text((cx, cy + T + 38), f"{r['수집처']} · {r['공정']} · 불량 확률 {r['불량확률'] * 100:.1f}%", font=_font(17), fill='#374151')
        m = re.search(r'_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_', str(r['이미지']))
        shot = f'{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}' if m else ''
        d.text((cx, cy + T + 62), f"촬영 {shot} · 확신도 {r['확신도']:.2f}", font=_font(15), fill='#6b7280')
    sheet.save(path)
    sheet.save(path[:-4] + '.pdf', 'PDF', resolution=150)     # 같은 내용의 PDF (내려받아 보관 · 인쇄용)
    return path


def view_url():
    return f'{SERVER}/#/site/{SITE}/views/SMT_/AI'


# ───────── 슬랙 ─────────
def send_slack_bot(text, files):
    """채널에 메시지 + 파일 (PDF · 워드) · SLACK_BOT_TOKEN (xoxb-) · SLACK_CHANNEL (이름 또는 ID)"""
    token, ch = uenv('SLACK_BOT_TOKEN'), uenv('SLACK_CHANNEL')
    if not (token and ch):
        return False
    token, ch = token.strip().strip('"\''), ch.strip().strip('"\'')
    if not (token.isascii() and token.startswith('xox')):   # 예시 글자(「실제토큰」)가 그대로 들어간 경우 등
        raise ValueError('SLACK_BOT_TOKEN 값이 실제 토큰이 아님 → Secrets 에 xoxb- 로 시작하는 봇 토큰을 넣으세요')
    from slack_sdk import WebClient
    c = WebClient(token=token)
    ch_id = ch
    if not re.fullmatch(r'[CG][A-Z0-9]{8,}', ch):      # 채널 ID 가 아니면 이름으로 찾는다
        name = ch.lstrip('#')
        for page in c.conversations_list(types='public_channel', limit=200):
            hit = [x['id'] for x in page['channels'] if x['name'] == name]
            if hit:
                ch_id = hit[0]
                break
    c.chat_postMessage(channel=ch_id, text=text)
    for f in files:
        c.files_upload_v2(channel=ch_id, file=f, filename=os.path.basename(f), title=os.path.basename(f))
    return True


# ───────── 명령 ─────────
def cmd_check():
    srv = sign_in()
    print('태블로 최종 유출 : 시작 상태', read_count(srv, base_hours()), '· 반영 후', read_count(srv))
    srv.auth.sign_out()


def cmd_push(threshold_new=1, notify=True):
    os.makedirs(R.OUT_DIR, exist_ok=True)
    if not os.environ.get('OPENAI_API_KEY') and uenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = uenv('OPENAI_API_KEY')
    t0 = time.time()
    df = load_full()
    batch = df[df['시간대'] == BATCH]
    srv = sign_in()
    before = read_count(srv, base_hours())
    print(f'[1] 반영 전 태블로 최종 유출 : {before}건')
    after = read_count(srv)
    new = after - before
    print(f'[2] 새 검사 묶음 반영 (촬영 {BATCH} · {len(batch):,}박스) → 태블로 최종 유출 : {after}건')
    print(f'[3] 기준 확인 : 새 유출 {new}건 (기준 {threshold_new}건 이상)')
    if new < threshold_new:
        print('기준 미달 : 알림 없음')
        srv.auth.sign_out()
        return {'before': before, 'after': after, 'new': new}
    leak = batch[(batch['오류종류'] == 'FN') & (batch['처리구분'] == 'AI 판정')]
    items = ' · '.join(f'{i} {c}건' for i, c in leak['검사항목'].value_counts().items())
    stamp = datetime.datetime.now()
    alert = {'시각': stamp.strftime('%m-%d %H:%M'), '새 검사 묶음': f'촬영 {BATCH} · {len(batch):,}박스',
             '새 유출': f'{new}건 ({items})', '태블로 최종 유출': f'{before} → {after}건'}
    srv.auth.sign_out()
    png = leak_sheet(leak.sort_values('불량확률'), os.path.join(R.OUT_DIR, f'유출사례_{stamp:%Y%m%d_%H%M%S}.png'),
                     f'유출 사례 {new}건 · 새 검사 묶음 촬영 {BATCH} ({items})')
    pdf = png[:-4] + '.pdf'
    print(f'[4] 유출 사례 PDF (실제 검사 사진) : {os.path.basename(pdf)}')
    k = R.compute_kpi()
    text, src = R.write_text(k, alert)
    dpath = os.path.join(R.OUT_DIR, f'AI판정검증_조치보고서_{stamp:%Y%m%d_%H%M%S}.docx')
    open(dpath, 'wb').write(R.build_docx(k, text, open(png, 'rb').read(), alert, img_title='유출 사례 (실제 검사 사진)'))
    print(f'[5] 워드 보고서 : {os.path.basename(dpath)} (문장 {src})')
    # <!channel> = 채널 전원에게 알림(휴대폰 · PC 팝업) · 그림은 슬랙에서 바로 펼쳐지고, 워드는 내려받는 보고서
    msg = (f"<!channel> 🚨 *SMT 유출 경보* · {alert['시각']}\n"
           f"새 검사 묶음 {alert['새 검사 묶음']}에서 *유출 {alert['새 유출']}*\n"
           f"태블로 Cloud 재계산 : 최종 유출 {alert['태블로 최종 유출']} (기준 : 새 유출 {threshold_new}건 이상)\n"
           f"첨부 PDF = 유출된 부위의 실제 검사 사진 · 워드 = 조치 보고서\n대시보드 열기 : {view_url()}")
    sl = send_slack_bot(msg, [pdf, dpath]) if notify else False
    print(f'[6] 슬랙 {"보냄" if sl else "안 보냄"} · 총 {time.time() - t0:.0f}초')
    return {'before': before, 'after': after, 'new': new, 'png': png, 'pdf': pdf, 'docx': dpath, 'slack': sl, 'alert': alert}


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if arg == 'push':
        cmd_push(notify='--no-slack' not in sys.argv)
    else:
        cmd_check()
